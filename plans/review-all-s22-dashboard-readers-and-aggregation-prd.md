# S22 — Dashboard readers and aggregation, and the cross-package runs.db read model (PRD)

**Status:** authored 2026-11-11 by the S22 seat of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S22, agent mode), with the critic's review folded
the same day; the rulings of the sitting of 2026-11-12 folded and re-checked against main on 2026-11-12
(§Rulings). Ruling R19 governs; R36 routes one more deliverable here (the cross-package `runs.db` read model;
lead addendum). B+H: contracts C1–C5 and a two-way boundary-test sketch (G5 fires on every arm: blast radius
dashboard, shared, orchestrator, fused-memory, client-feed and scripts; 33 leaves; three load-bearing seams).
Sized L by the R36 addendum (the synthesis sized the dashboard half M).

**Code anchors** verified against main `1b106418da` (2026-11-12). Main moves fast — cite-by-symbol;
re-locate lines at implementation time. Counts were measured at `ba17934861` (2026-11-10), each with its rule;
`git log ba17934861..1b106418da` (710 commits) changed 25 of this plan's files, and every count on a changed
file was re-measured and is labelled `measured at 1b106418da`. A count still labelled `ba17934861` sits on a
file no commit touched since. The run's anchors were verified at `6551b75aba`; every key was re-read at main
before a leaf was built on it (§Findings disposition). Re-checked at `0f21f5e457` (2026-11-12) by the fold
verifier: the 40 commits since `1b106418da` change no file of this plan, so no count moved. Aligned with the
sibling drafts and the lead's later rulings on 2026-11-13 (wave-3 alignment pass): edges that rest only on a test
file dropped, the S19 κ2 same-file edges completed, the decision-partition invariant cited by its slug.

## Goal

The dashboard reads every store it shows through one typed seam per producer, and every panel reports an
unreadable or unreachable source as unknown, never as zero. What a user observes when this lands:

- Stop fused-memory (`systemctl --user stop fused-memory`). The Recon, Memory-ops and Curator panels render
  "—" and name fused-memory as unreachable. The curator tab no longer reads "0 pending". The next metrics
  tick writes `NULL`, not `0`, into `curator_snapshots.pending_total` and `capped_now`.
- Make one project's `runs.db` unreadable. Its Spend (today) tile and topbar read "—", not "$0.00". Its
  performance cards and its history series both read unknown, by one rule. A project whose `runs.db` predates a
  column (an older orchestrator in the fleet) still serves every panel that does not need that column.
- Restart one orchestrator mid-request. The merge-queue endpoint still returns 200, with that project's live
  queue unknown and every other project's populated. Today an `httpx.ReadError` blanks every project's panel.
- No non-test module outside the read model holds SQL text naming `events`, `task_results`, `invocations`,
  `account_events` or `flake_occurrence`, and none pairs any of runs.db's 12 tables in SQL text with the
  runs.db locator. A guard test over every workspace member and `scripts/` fails if one appears. The
  dashboard (the quality markers included), fused-memory's `get_scheduler_events`, the client feed, the
  quality series and 14 scripts read `runs.db` through `shared.runs_read`. An orchestrator schema change that breaks one of those reads fails the
  orchestrator's own test suite.
- The merge-queue doughnut colours every live `OutcomeKind` by family. Today five live members render in
  the neutral "unknown" ramp. The family table is keyed by the imported `OutcomeKind` members and a test
  holds it total; the dashboard spells no event type or outcome string.
- An agent changing a dashboard reader finds, for each kind of change, one place to make it:
  - a fan-out site: one failure classification and one escalation-URL probe;
  - a cache: one cache module, whose per-key state is one record, and whose policies tests construct;
  - a route: one router module, taking a typed body;
  - a task row: it carries its raw `task_id`, and no reader splits the uid on `/T-`;
  - the scheduler: one named result record;
  - orchestrator discovery: one function.
- A merge that changes a dashboard asset no longer bumps a `?v=` literal on 50 tags. The browser revalidates.
- No dashboard test checks the browser app by reading its source text. The 30 test modules that do so through
  the source-text helpers today, and the tests outside them that read script text directly or run a sliced-out
  function through `_lock_chip_matrix.py` (D14 *Scope*), are render or node tests on the jsdom harness
  (`dashboard/tests/js/dom_harness.mjs`); the helper layer is deleted, and a guard refuses a new test that reads
  `static/redux` script text.

## Background

The run `review-all-dark_factory-20261008` grouped 22 dashboard findings into S22. R19 ruled the stream (option 1:
fused-memory's MCP status tools publish the panel read model; S15 owns those tools, S22 consumes). R36 (2026-11-10)
encapsulated the orchestrator's own event reads in S8 (tasks 9230 ι, 9231 ι2) and routed the readers outside the
orchestrator process to S22 as a numbered contract. The seam table gives S22 three seams: the label→root
resolver, one fan-out classification plus one `TTLCache` owner, and the cross-package `runs.db` events read model.

Two of the 22 keys were fixed on main after the run: task 5760's `ProjectLabels`, and the deletion of the
duplicate task-census walker. One ruled item had already landed: `data/retention.py::apply_retention` is the
one retention SQL home (task 6606, `ae4cdb038d`). The
rest are present at main, several larger than at the run. Examples: 54 script tags at `?v=109` against 46 at
`?v=77`; 10 live `TTLCache` instances against 8; 13 "LOAD CONTRACT" headers against 10.

Two concurrent wave-3 plans and two live non-program PRDs share this stream's files. The package-layering plan
(S19 C8, `dashboard/data/mcp_client.py`) owns the dashboard's MCP egress module, and S22's fan-out leaves build on
it. The CLI-invocation plan (S20 C7) makes `invocations` rows per attempt, so every `invocations` query in
the read model states whether it counts rows or sums them. The quality-dashboard PRD (`plans/quality-dashboard-prd.md`)
and the instruments-honesty PRD (`plans/instruments-honesty-prd.md`) have tasks that add runs.db readers and
edit S22's files; §Cross-PRD orders against each. One of them landed since the draft: quality-dashboard task
8460 (`5a1f289e91`) added `dashboard/data/quality_markers.py` with 4 raw runs.db SQL sites, which ρ3 moves onto
the read model.

Program rulings applied without relitigating: R19, R36, R22 (S22 owns no invariant), R23 (the four weakened
keys are re-verified below), R54/R55 (the wave-2 per-file order, which this PRD extends by explicit edges; test
files are left to the scheduler's file locks, per R54). The sitting of 2026-11-12 ruled this plan's eight
questions (§Rulings).

## Findings disposition

Measured at ba17934861 unless a row says `measured at 1b106418da` (a file changed since). Every key appears once. "Closes" names the leaf whose landing removes the cost; other
leaves that contribute are named in the note.

| Key | Sev | Disposition | Note (re-verified at main) |
|---|---|---|---|
| fk-a9e45ca0eb10 (weakened) | medium | leaf η | Kept under G3/G6. The skeptic's points hold: completion paths and escalation rates both sum then recompute, and percentiles can only be merged by concatenating samples. The cost that survives: `_merge_history` takes an unweighted mean of per-DB percentiles and has 0 direct tests and 0 multi-DB callers in tests. Cards and history follow two failure policies (cards: `_drop_partial_tallies` → UNKNOWN; history: `except Exception` → DEBUG, empty series). `get_time_centiles` has 0 production callers. The label-mismatch branch of `aggregate_loop_histograms` is reached only by 4 monkeypatches of `get_loop_histograms`. |
| fk-389c36ad02d1 | medium | fixed:7fc275667e | Task 5760 series. `dashboard/project_labels.py::ProjectLabels` makes labels unique across configured roots, so construction refuses a label naming two roots. `project_roots_for_label` is retired (7fc275667e). scheduler's `label_to_root` is gone (919a51fc83; results paired to roots by position). `_discover_escalation_urls` keys by `ProjectLabels`. Covered by `test_same_basename_roots.py`. Residue without a key: `redux_api.py::shape_performance` and `::shape_costs` label a runs.db project_id with `basename_label`. η and ι label by the DB's root through C1. |
| fk-70e1ee9218a7 | medium | leaf κ | Prose drift. Fixed since the run: the `data.js` mock-fixture claim, `collect_done_counts`, `_analytics_project_dirs`. Survives:<br>• mcp_fanout's "8 module-level instances spread over 4 modules" (true: 10 instances, 8 modules), "~8 live TTLCache instances", "the three tasks sites" (true: 5), and merge_queue as an imminent consumer (it imports only `describe_exc`);<br>• active_tasks' 66-line `_TASKS_TOTAL_BUDGET` block, which still holds the 2026-09-07 and 2026-09-23 campaigns;<br>• tasks.py's consumer list (3 of 4 stale, measured at 1b106418da: `orchestrator`, `burndown` and `merge_queue` import nothing from it). Its "NO field/column projection" claim was fixed by task 9311 (`b88ebc7dda`);<br>• scheduler's "does not exist yet" rationale (`event_store.py` defines `recovery_vetoed`).<br>Prose share: mcp_fanout 0.727, active_tasks 0.616, scheduler 0.494. γ1 fixes tasks.py, δ1 fixes mcp_fanout, κ fixes active_tasks and scheduler and closes the key. |
| fk-1c1d247caaa7 (weakened) | medium | leaf ε | Kept, in the narrower form. 11 SQL sites over 8 fused-memory tables (rule: `(FROM|JOIN)\s+<table>` over dashboard/src ∩ fused-memory `CREATE TABLE` names):<br>• `reconciliation.py`: runs 3, watermarks 1, event_buffer 2, burst_state 1, judge_verdicts 1;<br>• `write_journal.py`: write_ops 1, read_op_counts 1;<br>• `metrics.py`: tickets 1.<br>`journal_entries` went with task 5890, so the run's 13 sites are 11 at main, as S15 C4 now pins. The `INDEXED BY idx_wo_created` hint and its missing-object retry loop survive. 8 fused-memory DDLs are hand-copied in 4 test files. Table and column drift is still unguarded. ζ removes the tickets site; ε closes. |
| fk-b5760d0dfe85 | high | leaf δ1 | The five per-key structures are still `_store`, `_locks`, `_bypass_streaks` (now `KeyedStreaks`), `_bypass_tasks` and `_live_bypasses`, reconciled by four reclaimers plus `clear`. `_evict_expired` still reads `asyncio.Lock._waiters`. TTLCache spans lines 704–1548 of `mcp_fanout.py`, with prose share 0.685. Tests reach the cache's privates: `_locks` 27, `_bypass_tasks` 22, `_live_bypasses` 13, `_store` 8, and `setattr` of `_LOCK_ACQUIRE_TIMEOUT_SECONDS` 32 times. 12 test files patch a module TTL constant (rule: `git grep -l -E "setattr\([^)]*_TTL_SECONDS\|_TTL_SECONDS', " -- dashboard/tests`); δ2 moves them onto constructed policies. |
| fk-94fc76e7a123 | high | leaf ξ2 | Measured at 1b106418da: `index.html` has 54 `<script>` tags; 49 carry `?v=109` (50 occurrences with `styles.css`); 0 are `type="module"`. 17 tags are `text/babel` compiled in the browser by Babel standalone 7.29.0. 13 files carry a LOAD CONTRACT header. 39 commits since the run changed a `?v=` line (rule: `git log --oneline -G'\?v=[0-9]' 6551b75aba..main -- dashboard/src/dashboard/static/redux/index.html`). 18 files reference `?v=` or `_cache_buster_helpers` (rule: `git grep -l -E '_cache_buster_helpers\|\?v=' -- dashboard/tests dashboard/src`, vendor excluded). ξ1 serves revalidation and retires the floor pins; ξ2 removes the cache-buster invariant. The scope and load-order invariants stay enforced by `classic_script_scope.test.mjs` and `test_index_html.py` (ruled at the sitting, OQ-3). |
| fk-8dfa5e5a75aa | medium | fixed:c3ed0f677b (the duplicate walker); residue — client-side completeness over non-snapshot get_statuses paging — owned by 5717 (docstring) and 3396 (pagination) | `_walk_pages` was deleted by c3ed0f677b (authored 2026-10-13, committed 2026-10-17; `fetch_tasks` is one call). The finding's cost was two walkers sharing a failure model but not code, so a fix reached one and not the other; that cost is gone. One walker, `dashboard/data/tasks.py::_walk_statuses`, still proves completeness client-side over a non-snapshot paged tool; the PRD view, deps and readiness need its per-task status map (`task_snapshot.status_map` has 4 consumer modules). Live tasks 5717 (the upstream docstring) and 3396 ("Defensive pagination handling in the two programmatic get_statuses consumers") own that residue. |
| fk-c927adb040e9 | medium | leaf θ | `outcome_colors._CODE_FAMILY` (no commits since the run) has 29 keys. 11 are not emitted (`cas_failed`, `not_descendant`, `stash_failed`, …). 5 live `OutcomeKind` members are missing (`main_health_red`, `plan_files_cross_repo`, `plan_files_already_landed`, `gate_retry`, `post_merge_generation_chained`). The header cites the deleted `merge_queue.py:2074-2080`. α moves the vocabulary; θ keys the family table by the imported members, held total by a test. |
| fk-1b8c06cd8b2f | medium | leaf ν | `app.py` is 1,708 lines (measured at 1b106418da), prose share 0.425 at ba17934861. It registers 16 `@app` routes beside 10 `include_router` calls (the 10th, the quality router, landed with task 8459). The 5 write proxies hand-validate bodies with `isinstance` checks; 0 use pydantic. `_VALID_BOOST_TIERS` restates `PRIORITY_TIERS`. `_VALID_CLEAR_FIELDS` uses `ttl_until` where fused-memory uses `ttl`. Live task 8700 overlaps. |
| fk-ce66d4da9b0f | medium | leaf ν | Same file. Heuristic-14 partition, with no reach-back for any candidate (measured at 1b106418da; `git diff ba17934861..1b106418da -- dashboard/src/dashboard/app.py` adds one line to the healthz fan-out probe and 17 to the clear-override proxy, task 4867 `17b97a931e`):<br>• healthz: 468 lines;<br>• GET endpoints: 423;<br>• write proxies: 338;<br>• lifespan and limits: 314;<br>• wiring: 35. |
| fk-ad49b80c5554 | high | leaf γ2 | All four probes are present, each spelling `removesuffix('/mcp')` (4 sites). They differ in three ways:<br>• exception tuples: merge_halt uses `(*FANOUT_FAILURE_EXCEPTIONS, OSError)`; merge_queue and escalations use `(TimeoutError, httpx.HTTPError, OSError, ValueError)`; task_runtime uses deadline-first `TransportError`;<br>• `gather` mode: only escalations isolates;<br>• log level: three log at DEBUG only.<br>Live defect, verified with httpx 0.28.1: `httpx.ReadError` escapes `merge_halt._probe_one`, and `api/merge_queue.py::api_merge_queue` still bare-gathers, so it becomes a 500. `merge_halt` calls `result.get` with no dict guard. |
| fk-af4ba730a48a | high | leaf θ | `merge_queue.py` holds 9 `events` sites with 7 event-type literals, plus `$.outcome`/`$.train_id` paths and the outcome strings `'done'`/`'cas_retry'`. `static/redux/tabs.jsx` spells `'train_merged'` for the event badge. Measured drift against `EventType`: 0. The cost is the restatement (11 dashboard literals and 4 more via MCP). Hand-copied runs.db DDL: 29 hits in 16 dashboard test files (measured at 1b106418da; task 8460 added `runs` and `invocations` DDL to `_quality_fixture.py`), `MERGE_EVENTS_SCHEMA` and `_runs_db_fixture.py` included. |
| fk-4afbeb684a22 | high | leaf ζ | All five pieces present:<br>• `fan_out_list_tickets` `offline_result=lambda errs: (0, [])`;<br>• `api_curator` `safe_gather_result(..., ([], 0))`, with a `{'offline': True}` marker that is never surfaced;<br>• `shape_curator` has no unknown field;<br>• `compute_capped_now_and_windows` failure → 0 in both `api_curator` and `_sample_curator`;<br>• `_sample_curator` writes 0 into metrics.db.<br>Live task 7815 has the same class on the memory and queue samplers. |
| fk-166cacf84a18 | medium | leaf μ | `_mcp_probe` and `_mcp_fanout_last_ok` are mutated through `global`, and the `_mcp_probe_state_clear` hook is still present. The budget relation is checked only by `test_healthz_deadline.py`. 7 verdict strings are spread over membership tests. 22 distinct app.py private names are reached from dashboard/tests, 11 of them healthz names (rule: attribute access, `setattr(app_module, '_X'` or `from dashboard.app import _X`). |
| fk-d21f64525e52 | medium | leaf κ | Producers: `active_tasks.py::task_uid`, the `task_prose` route template and `prd_lanes_view.js`. Parsers: `scheduler.py` `rsplit('/T-', 1)`, `scheduler_utils.jsx`, `tabs.jsx` and `shell.jsx`. The row (`active_tasks.py`, 19 keys) has no `task_id`. Events are keyed `f'{label}/{tid}'`. |
| fk-9979f1b771de | medium | leaf ι | `db.py::with_db` is unchanged. It has 28 call sites: costs 7, merge_queue 6, reconciliation 6, performance 5, cap_history 1, load 1, model_role 1, plus the definition. Its defaults are `{}` / `[]`. `redux_api.py::shape_costs` still serves `'today': … else 0.0`, which `spend_readings.js::plainDatum` renders as a fresh $0.00. |
| fk-c51423b7b7e5 | medium | leaf δ1 | `mcp_fanout.py` is 1,548 lines (prose 0.727) holding both mechanisms. The function-local `from dashboard.data.memory import invalidate_session` inside `first_success` breaks a cycle. `data/burndown.py` imports the private `_LOCK_ACQUIRE_TIMEOUT_SECONDS`. There are 22 task-number citations. The cycle is closed by S19 κ1 (fk-99f44f47a6cc, S19 C8); δ1 closes the transport/cache split. |
| fk-d0a7ebc4a4f1 | medium | leaf λ | Three process→root resolutions disagree on a bare `run` or an unreadable `--config`:<br>• `orchestrator.py::discover_orchestrators._resolve_root` attributes it to the default root;<br>• `metrics.py::_sample_orchestrators` counts it there;<br>• `burndown.py::collect_snapshot` skips it.<br>`ps aux` runs with no `timeout=` while a `TimeoutExpired` clause catches nothing. `config.py::_read_escalation_url` returns None without `escalation.port`; `EscalationConfig.port` defaults to 8100. Live task 6875 overlaps. |
| fk-6aee6c49fa2e | medium | leaf κ | `collect_scheduler_state` spans 338 lines with 3 nested functions. It returns a positional 8-tuple (it was 7), which `app.py::api_scheduler` unpacks by position. Cognitive complexity not measured (complexipy absent from the venv). Live task 6842 overlaps. |
| fk-f3a5a34831e9 (weakened) | medium | leaf γ1 | Kept, as the residue. The failover is consolidated (10 `first_success` sites). Still present:<br>• `memory.py::get_queue_stats` and `::get_wal_status` loop over URLs and sum by hand, against `DESIGN.md`'s failover-only topology (live task 8119);<br>• `metrics.fan_out_list_tickets._call` keeps its own three-step except chain;<br>• `FANOUT_FAILURE_EXCEPTIONS` misses the `TransportError` family (live task 6872);<br>• `task_lineage.py` hand-rolls a second single-flight TTL cache, which δ1 moves onto C3. |
| fk-1bdd95e804cb | medium | leaf π8 (ο guards; π1–π7 convert) | Measured at 1b106418da with the rule `git grep -l -E 'strip_js_comments\|extract_function_body\|app_jsx_body\|_app_jsx_source\|jsx_prop_expr' -- dashboard/tests`, then `grep -c '^\s*def test_'` per `test_*.py` hit: 37 files, 33 of them test modules holding 610 tests (614 at ba17934861). Three of the 33 are not conversion work: `test_jsx_source_helpers.py` (109 tests) is deleted with the helpers, `test_cache_buster_freshness.py` (31) goes with ξ2, and `test_index_html.py` (73) keeps its structure tests (OQ-3) with one served-script-text test converting. The conversion population is 30 modules holding 397 tests, about 15,000 lines. Python tests outside that rule that C6's guard also finds are converted too (D14 *Scope*, re-measured at 0f21f5e457): `test_charts_consumer_bindings.py` (4 tests), the `DEFAULT_TIMEOUT_MS` read in `test_prds_budget.py` and `test_tasks_budget.py` (one test each), `test_datum_access_paths.py::test_no_served_script_binds_a_retired_client_name`, the served-module body checks in `test_tab_tasks_deferral.py`, `test_tab_tasks_invariant_chip.py`, `test_tab_tasks_offline_banner.py` and `test_index_html.py::test_scheduler_heatmap_bounds_js_is_served`, conftest's nine served-script fixtures, and the two importers of `_lock_chip_matrix.py` (`test_lock_chip_state.py`, 5 tests; `test_boundary_js.py`'s sketch-12 cell-and-chip test). The jsdom harness renders any tab (`tests/js/dom_harness.mjs::buildDashboardWindow`; `jsx_render.test.mjs` ran 79 tests in 9.7 s on 2026-11-10). Ruled at the sitting (OQ-4, option B): ο stops new ones; π1–π7 convert by tab; π8 deletes the helper layer and closes the key. |
| fk-9cf2a161c9c0 (weakened) | medium | leaf λ | Kept for parts (a) and (b): the `_expand_env_placeholders` regex copy, which skips `config.py`'s bool and None coercion, and the `ps aux` regex discovery. Part (c), the `.task/` artifact scraping, was retired by 3f78c8a09c. |

## Pre-conditions for activating

- **S14 κ 6722** (pending; it depends on 6718 and 6719, both pending) changes the dashboard's DB-path
  literals, its MCP session and its error codes, in the files γ1, γ2, δ1, μ, ε and λ also edit. Those leaves
  depend on 6722. The G4 row is "`shared.project_layout` for DB paths (S14)".
- **S19 κ1 and κ2** (unfiled; draft `plans/review-all-s19-package-layering-and-private-seams-prd.md`, contract
  **C8**: `dashboard/data/mcp_client.py`, the dashboard's only MCP egress, holding `mcp_tool_call`,
  `invalidate_session`, `reset_sessions` and `enclosing_deadline`) break the `mcp_fanout`↔`memory` cycle and make
  the dashboard's cross-module underscore names public. γ1, γ2 and δ1 depend on κ1; γ1, γ2, δ1, μ, ν and λ on
  κ2 (lead ruling S19/Q-C3 = S22/OQ-6; S19's κ2 states that this plan's fan-out leaves and app.py partition
  depend on it, and each of these leaves shares a non-test file with κ2).
- **S15 κ** (unfiled; draft `plans/review-all-s15-fused-memory-mcp-server-boundary-prd.md`, contract **C4**:
  `shared/src/shared/dashboard_panels.py` reply models served by `get_status(panels=…)`) gates ε and ζ.
  The lead's S15/OQ-8 puts κ first on its tools.py track, right after 9280 and S19 α.
- **S16 α** (unfiled; draft `plans/review-all-s16-orchestrator-config-shape-defaults-and-routing-prd.md`,
  contract **C8**: `shared/src/shared/config_defaults.py`, the dependency-free cross-package defaults) gates λ,
  which adds the escalation port to it (lead ruling S16/OQ-4 option 2 = S22/OQ-5).
- **S20 ζ** (unfiled; draft `plans/review-all-s20-cli-invocation-and-usage-gate-prd.md`, contract **C7**:
  `invocations` rows per attempt with `invocation_id`, `attempt`, `returned`) gates ι, ρ1, τ and υ, whose
  `invocations` queries must encode its sum-only / row-as-invocation classification.
- **S6 α 9140** heads every wave-2 chain on `merge_lane/types.py` (R54), so α follows it. Task **7612** landed
  (`3b1f243a9b`): the script's copy is now the public `NON_TERMINAL_OUTCOMES`, and
  `orchestrator/tests/test_outcome_kind.py::TestOutcomeKindFrozenContract` loads the script by path and fails if
  the two sets differ.
- **S3 γ 9162** repoints imports and prose in `dashboard/data/escalations.py`, `performance.py`,
  `escalation_corpus.py` and `scripts/sitting/fleet_state.py`. γ2, δ1, η and τ follow it.
- **S8 ι2 9231** (depends on 8563 and 9230) gates υ.
- **Instruments-honesty and quality-dashboard tasks** 8461 (in progress), 8558 (in progress), 8559, 8563 and
  8465 edit S22's files and are older; the S22 leaf on the same file follows them (§Cross-PRD). 8460 landed
  (`5a1f289e91`) with raw runs.db reads; ρ3 moves them. 8561 and 8564 are parked behind 8528, so they follow the
  S22 leaf on the same file instead (8561 → γ2; 8564 → δ1, ι).
- Substrate verified present (G3), each measured at ba17934861:
  - `shared/project_layout.py::runs_db`;
  - `shared/mcp_post.py::mcp_server_base` (task 6747);
  - `shared/task_transitions.py::WORKFLOW_OUTCOMES`;
  - `shared/cost_store.py` (it owns the invocations and account_events DDL);
  - `shared/tasks_db_readonly.py::connect_ro`, `TaskDbProblem`, `TaskDbUnreadable` (the read-only opener five
    runs.db readers already use);
  - `shared/ratchet.py` (the INV-12 ratchet kernel) and `shared/governed_exceptions.py::governed_exceptions`,
    `Debt`, `TaskRef` (the INV-12 disposition vocabulary; precedent
    `dashboard/tests/test_dashboard_governed_lists.py`);
  - `dashboard/data/datum.py`: `Datum`, `unknown_datum`, `DatumState`;
  - `dashboard/project_labels.py::ProjectLabels`;
  - `shared/delivered_check_polarity.py::lint_delivered_checks`;
  - FastAPI `StaticFiles`, which sends ETag and Last-Modified and answers a conditional GET with 304;
  - a node and jsdom harness under `dashboard/tests/js/`, whose `_served_bundle.mjs` tag regexes already accept an
    unstamped tag (`(?:\?[^"]*)?`); `dom_harness.mjs::buildDashboardWindow` loads every local script in
    index.html order and returns `mount(Component, props)` and `mountPage()`;
  - `dashboard/tests/test_graph_layout_js.py`, the pytest wrapper that runs every
    `dashboard/tests/js/**/*.test.mjs` (`_JS_TESTS_GLOB`), so a conversion task's new render-test file needs no
    registration.

## Sketch of approach

Six tracks, serialized on shared files by real edges ("X → Y" means X depends on Y). α and β come first, the
dashboard's app.py chain second; the rest order by file.

1. **Vocabulary and read model** (β → α; ρ1, ρ2, ρ3, σ, τ, θ, η, ι → β; υ last). α moves `EventType` and
   `OutcomeKind` into `shared` (C4). β builds `shared.runs_read` (C5): queries as values, `Read | Absent |
   Unreadable` results, the shared read-only opener, an executor on a caller's connection, a schema-skew
   rule, a fixture writer, and an orchestrator-side contract test over a DB the real stores wrote and over the
   fleet's oldest schema. ρ1, ρ2, ρ3, σ, τ, θ, η and ι move each reader family onto a query module of its own.
   ω's guard then refuses any new raw read. υ makes `EventStore`'s typed reads (S8 ι2) delegate their SQL to
   the same queries, so the digest and the dashboard share one model-role query.
2. **Fan-out and cache** (γ2 → γ1; δ1 → γ1; δ2 → δ1). They build on S19 C8's egress module. γ1 fixes one
   failure classification and puts memory's queue and WAL loops on `first_success` (C2a). γ2 collapses the four
   escalation-URL probes onto one per-label-isolated helper (C2b). δ1 moves `TTLCache` into its own module with one
   slot record per key and a construction-time policy (C3), and puts `task_lineage`'s hand-rolled single flight on
   it. δ2 makes the module caches app-scoped behind one registry, so tests construct policies instead of patching
   TTL constants.
3. **app.py** (μ → 6722, ν → μ, ξ1 → ν, ξ2 → ξ1). μ moves the healthz apparatus into a `HealthProbe` held on
   `app.state`. ν moves the remaining routes into `api/` routers with pydantic bodies, so `app.py` keeps
   construction, lifespan and router wiring. ξ1 makes `/static` revalidate and retires the tests that pin a `?v=`
   floor; ξ2 strips the stamps and deletes the freshness test.
4. **Panels honest about outages** (ζ, ε, ι). ζ makes the curator panel and the metrics samplers carry unknown.
   ε reads the recon and memory-ops panels from S15's read model and deletes the dashboard's fused-memory SQL
   and DDL. ι converts the last `with_db` callers and deletes `with_db`.
5. **Shapes** (κ, λ, ο). κ turns the scheduler collector into fetch plus a pure shaper returning a named record,
   with a raw `task_id` on every row. λ gives orchestrator discovery and the orchestrator-yaml read one function
   each.
6. **Source-text tests become render tests** (ο; π1–π7; π8 last; D14). ο guards against a new test that reads
   frontend source text, with an exemption list in which every module is owned by its conversion task. π1–π7
   each rewrite one tab group's source-text modules as jsdom render tests in a `.test.mjs` file of their own, so
   no chain runs among them; each waits only for the plan tasks that rewrite its tab or its test modules. π8
   converts the last served-script reads, deletes the helper layer, and leaves the exemption list empty.

ω is the integration gate and signal leaf: the boundary rows end to end, plus the runs.db encapsulation guard. ω
waits on ο but not on π1–π8, so the medium conversion tasks never become prerequisites of gate 6698.

## Resolved design decisions

- **D1 — The label→root resolver is `ProjectLabels`, as task 5760 delivered it (C1).** R19 named the resolver's
  result `one|unknown|ambiguous`. Since `ProjectLabels.__post_init__` refuses a label naming two roots,
  ambiguity is unrepresentable after construction, and `root_for` returns `Path | None`. S22 adopts it as the
  seam it owns and files no resolver leaf. η and ι label runs.db tallies by the DB's root through C1, not by
  `basename_label(project_id)`.
- **D2 — The run vocabulary moves to `shared`; the orchestrator's defining modules import it (C4). Ruled at
  the sitting (OQ-1): move now, with the full amend set.**
  R19 rules the vocabularies "imported not re-spelled". `shared.runs_read` must filter on them, and
  fused-memory, the client feed and the shared quality series cannot import `orchestrator`. So `EventType`
  (130 members, measured at 1b106418da; task 3729 added `sidecar_reaped` and `sidecar_reap_summary`) goes to
  `shared/run_events.py`, and `OutcomeKind` (23 members) and `NON_TERMINAL_OUTCOMES` go to
  `shared/merge_outcomes.py`. `event_store.py` and `merge_lane/types.py` import them. Their 46 and 4
  orchestrator importers are not codemodded, for two reasons. `EventStore.emit` takes an `EventType` and
  `MergeOutcome` carries an `OutcomeKind`, so each name is part of its module's interface, not a shim kept so an
  old path resolves (heuristic 13). And a 46-file codemod would contend with 20 wave-2 tasks' locks (measured at
  ba17934861 from `digests/wave2-order.json`: 8 on `event_store.py`, 16 on `types.py`, 3 in both, S6 α 9140
  ahead of α). The cost of the move is ongoing: 28 commits in the 60 days to 2026-11-10 added an `EventType`
  member, and 22 open tasks are expected to add a member (21 pending, 1 deferred, read 2026-11-12; the 23rd of
  the draft, 7612, landed as `3b1f243a9b` without adding one); after α each of them edits `shared/` instead. α
  adds a one-line pointer to `docs/task-authoring.md`, and the lead amends the 22 so each declares the shared
  file. The lead ruling on S16/OQ-4 (option 2) keeps the dashboard off the `orchestrator` package, so D2 is
  needed on that count too; the shared read model and the non-dashboard readers need the vocabulary below them
  regardless. The colour family of an outcome is the dashboard's
  concern, not the vocabulary's (heuristics 3 and 6): `dashboard/data/outcome_colors.py::OUTCOME_FAMILY` is keyed
  by the imported members, a test holds it total over `OutcomeKind`, and a recorded outcome string that is not a
  member (a retired code in an old event) is `family_of(...) is None`, which the doughnut renders as "retired".
- **D3 — The read model is a library of query values in `shared`, not a service (C5).** The readers include
  offline scripts, the quality-series CLI and the dashboard's history panels, all of which must work with the
  orchestrator down. R36 records that no in-process design gives them read-your-writes anyway. A `RunsQuery[T]`
  is `(sql, params, decode)` plus optional column requirements; the executors run it, so one SQL text serves
  `sqlite3` scripts, the dashboard's pooled `aiosqlite`, and a caller that runs several queries on one snapshot
  (heuristic 7: stateless; heuristic 11). Each reader family has its own query module, so the leaves that
  migrate different families do not edit one module (the quality markers that task 8460 landed get
  `runs_read/markers.py`, under ρ3). The guard against drift is producer-side, and it is an
  integration test, not a parity test. `orchestrator/tests/test_runs_read_contract.py` writes through the real
  `EventStore`, `RunStore`, `CostStore` and flake ledger, then runs every exported query (INV-13); it enumerates
  `shared.runs_read`'s exported query constructors, so a new export without a row fails it. An orchestrator
  migration that breaks a read therefore fails where the migration is made — the consequence R19 bought for
  fused-memory. `runs_read/fixture.py::write_runs_db_fixture` is the read model's fixture writer, checked against
  the real schema; the other six test-side runs.db DDL copies (`scripts/tests/conftest.py::make_runs_db`,
  `client-feed/tests/_stores.py`, `scripts/tests/test_sitting_return_brief.py`,
  `tests/scripts/test_census_tagger_debris.py`, `tests/scripts/test_repair_wiped_metadata_files.py`,
  `dashboard/tests/test_dashboard_endpoints_survive_hung_mcp.py`) migrate to it opportunistically.
  Two guards, two rules. ω's guard: no raw runs.db SQL outside the read model. The quality guard
  (`dashboard/tests/test_datum_access_paths.py`): per-landing quality facts are reached only through
  `data/quality_series.py` / `quality_sampler.py`, whether by SQL text or by `runs_read` query name, so θ and ρ1
  add the `runs_read` constructors that return per-landing quality sources to its `GUARDED_QUALITY_READS`
  (`plans/quality-dashboard-prd.md` decisions 14, 15, 22).
- **D4 — Three result states, at one place, through the one read-only opener.** `Read(value)`; `Absent(path)` when
  the file does not exist (no producer has written it yet); `Unreadable(path, reason, error_type)` for every other
  refusal, `sqlite3.Error` or `OSError`. A decode bug propagates (INV-11; INV-2). The executors open through
  `shared/tasks_db_readonly.py::connect_ro`, which five runs.db readers already use; β gives it a store-keyed
  remedy table (`store=SqliteStore.RUNS`), because its tasks.db remedy prose would mislead on runs.db (task 7835),
  and extracts its URI builder as `read_only_uri(path)`. `TaskDbProblem.ABSENT` maps to `Absent`; every other
  `TaskDbProblem` and `sqlite3.Error` maps to `Unreadable`. The async executor and task 8661's
  `async_sqlite_base.connect_read_only` build their URI with `read_only_uri`. The dashboard maps `Absent` and
  `Unreadable` to `unknown_datum` with the reason; with ι, `with_db` is deleted.
- **D4b — Schema skew is a value, not a blank panel.** The read model reads every project's runs.db across a fleet
  running mixed orchestrator versions (orchestrator-reify runs pre-6106 code). A `RunsQuery` may declare
  `requires_columns: Mapping[str, frozenset[str]]`. The executor probes `PRAGMA table_info` once per connection
  and returns `Unreadable(reason='schema older than <table>.<column>')` for a missing column, unless the query
  supplies a `fallback` variant (the `quality_series/rows.py::_has_charter_column` pattern, which ρ1 moves into
  this form). A query that does not need the column never fails on it. The contract test runs every query
  against the fleet's oldest supported schema (`orchestrator/tests/fixtures/runs_db_schema_floor.sql`, captured
  by β from the oldest live runs.db with `.schema`).
- **D5 — Panel outage semantics, one rule.** A panel whose source is unreachable (MCP), unreadable (SQLite) or
  partially read carries an UNKNOWN `Datum` with a reason and renders "—". A sampler that could not measure
  writes `NULL`. No shaper defaults a reading to 0 (ζ, ε, ι; `CONVENTIONS` "never synthetic data").
- **D6 — Fan-out: one classification, one probe, deadline first (C2).** `FANOUT_FAILURE_EXCEPTIONS` becomes
  `(httpx.TransportError, httpx.HTTPStatusError, TimeoutError, ValueError, OSError)`. `TimeoutException` is a
  `TransportError`, so the new tuple catches every member of the old one. `classify_fanout_failure` maps
  `TimeoutError`/`httpx.TimeoutException` → deadline, transport/OS/HTTP-status → unreachable, `ValueError` →
  bad_reply. The escalation-URL probe isolates per label and logs a WARNING on transition only, through
  `KeyedStreaks` (INV-4). `shared.mcp_post.mcp_server_base` derives base URLs.
- **D7 — The cycle is broken by S19 κ1 (S19 C8, `dashboard/data/mcp_client.py`). S22 consumes C8 (lead ruling
  S19/Q-C3 = S22/OQ-6).** C8 is
  the dashboard's only MCP egress (`mcp_tool_call`, `invalidate_session`, `reset_sessions`,
  `enclosing_deadline` and the per-URL session cache), below both `memory.py` and `mcp_fanout.py`. γ1, γ2 and δ1
  depend on κ1 and use its names; S22 creates no session module.
- **D8 — `TTLCache` takes a frozen `CachePolicy` at construction (δ1); the module caches are app-scoped and tests
  construct, never patch (δ2) (C3).** δ1 removes the `asyncio.Lock._waiters` read, since the slot counts its own
  waiters, and moves the lock timeout into `CachePolicy` (default `ttl_cache.py::DEFAULT_LOCK_ACQUIRE_TIMEOUT_SECONDS`);
  `key_state(key)` is the public test seam, and every reclaim path calls one `_reclaim(key)` decision. δ1 keeps the
  callable `ttl_seconds` so construction sites change by one line. δ2 builds every module cache in lifespan into
  one `CacheRegistry` held on `app.state.caches` and read through one accessor; the readers keep their signatures.
  With it go the callable-ttl form, which exists so a test can monkeypatch a module TTL constant, the module TTL
  constants, and every test `setattr` of one (12 files).
- **D9 — The healthz apparatus is a `HealthProbe` on `app.state` (μ).** It is constructed from a
  `HealthBudget`, whose per-probe timeouts are derived from the total. The relation is validated at
  construction and raises `ValueError`, naming the values (heuristic 10, at construction and at use). A
  `ProbeVerdict` StrEnum carries `is_healthy` and `is_deadline`. The module globals and the reset hook go.
- **D10 — Write proxies take pydantic bodies (`extra='forbid'`); `project_root` must be a configured root
  (ν).** FastAPI's 422 replaces the hand-built 400s. The boost tiers come from `shared.task_priority`
  (S14 θ/κ). The clear-field vocabulary maps `ttl_until` to fused-memory's `ttl` once, in the body model.
- **D11 — Static assets revalidate instead of busting (ξ1, ξ2). Ruled at the sitting (OQ-3): revalidation only;
  `classic_script_scope.test.mjs` and `test_index_html.py`'s structure tests stay the enforcement of the scope
  and load-order rules.** The dashboard serves from the main
  checkout, and merges change assets without restarting it. A version computed at startup would therefore go
  stale. ξ1 serves `/static/redux/*` and `/` with `Cache-Control: no-cache` plus `StaticFiles`' ETag, so every
  load revalidates (304 when unchanged), and retires the ten tests that pin a `?v=` floor or a sibling-version
  agreement. ξ2 then removes `?v=` from all 50 occurrences and deletes the freshness test and its helpers, and the
  Mem0 norm that tells agents to bump (topic `dashboard-redux-cache-buster`) is superseded.
- **D12 — Orchestrator discovery is one function returning typed records (λ).** It returns
  `RunningOrchestrator(pid, root: Path | Unresolved(reason))`, with `ps` under a timeout. The panel shows an
  unresolved process as unresolved. The sampler and burndown skip it, both by calling that one function.
  The dashboard reads orchestrator yaml through one reader that applies env expansion, with the same bool and
  None coercion as `config.py`, and the defaults, uniformly. The defaults come from S16's C8
  (`shared/src/shared/config_defaults.py`), as the lead ruled S16/OQ-4 (option 2) and S22/OQ-5 with it: the
  reader imports `DEFAULT_MAX_CONCURRENT_TASKS` (S16 α) and `DEFAULT_ESCALATION_PORT`, which λ adds to C8 while
  `orchestrator/src/orchestrator/config.py::EscalationConfig.port`'s field default comes to name it (C8's "a
  field default naming one shared constant is one home"). Nothing is restated, and the dashboard takes no
  dependency on `orchestrator`.
- **D13 — Prose (Comments stance).** Each leaf that rewrites a module moves its dated measurements, rejected
  alternatives and inventories into the commit message or task record, leaving at most a one-line pointer. A
  hand-kept inventory becomes a pointer to the mechanism that enumerates it (`_live_caches`, `git grep`).
- **D14 — Every test that reads frontend source text becomes a render test; a guard stops new ones (ο, π1–π8,
  C6). Ruled at the sitting (OQ-4, option B).** The existing modules are converted now, by tab, onto the jsdom
  harness (`dashboard/tests/js/dom_harness.mjs::buildDashboardWindow`, `mount`, `mountPage`), and the
  source-text helper layer goes with the last of them.
  - *Grouping.* Seven conversion tasks, one per tab group (π1–π7), and one cleanup task (π8). Each writes its
    render tests in a new `dashboard/tests/js/*.test.mjs` file of its own, which `test_graph_layout_js.py`'s
    glob already runs, so π1–π7 share no production file and need no chain among them. A render test asserts
    on rendered DOM and on the classic-script helpers' public globals, never on script text.
  - *Fidelity.* A source-text test can pin a detail with no visible effect in jsdom, such as a memo key or a
    hook call shape (`test_tab_tasks_layout_memo.py` pins memoisation). The conversion replaces such a pin by a
    behaviour check where one exists (a render count, a re-render on changed data) and otherwise drops it.
    Each task lists every pin it drops, with the deleted test's name and the reason no render test holds it,
    in its commit message.
  - *The guard and its list.* ο's guard fails on a `dashboard/tests` Python module that names a source-text
    helper or reads the text of a `static/redux` `.js`/`.jsx` file (C6). The modules that do so when it lands
    are on its exemption list, each owned by its live conversion task: the owned-debt arm of INV-12, so no
    ratification row is written. The guard compares what it measures with the list through
    `shared/src/shared/ratchet.py`, so a new offender fails (excess) and so does a converted module left on
    the list (slack): the list only falls. Each conversion task declares the list file and deletes its own
    modules from it; one that lands before ο simply never appears on it. π8 leaves the list empty and the file
    in place.
  - *Scope.* The ruled population, measured at 1b106418da: 30 modules holding 397 tests, about 15,000 lines,
    plus the one served-script-text test in `test_index_html.py` (§Findings disposition). The guard finds more
    than the helper rule does, and under the owned-debt rule every module it finds needs an owner. Measured at
    `0f21f5e457` (rule: C6's offender rule over `dashboard/tests/*.py`):
    - the direct-read clause finds `test_charts_consumer_bindings.py` (π3), the two `DEFAULT_TIMEOUT_MS` reads
      in `test_prds_budget.py` and `test_tasks_budget.py` (π7), the retired-client-name census in
      `test_datum_access_paths.py` (π8), and conftest's nine served-script fixtures (`data_js_body`,
      `app_jsx_body`, `shell_jsx_body`, `tabs_jsx_body`, `charts_jsx_body`, `tab_analytics_jsx_body`,
      `tab_overview_jsx_body`, `tab_escalations_jsx_body`, `tab_tasks_jsx_body`; every reader is a module
      π1–π8 delete, so π8 deletes them);
    - the same clause finds four served-module body checks (`'deferralCellState' in resp.text` in
      `test_tab_tasks_deferral.py`, `'invariantChipFor'` in `test_tab_tasks_invariant_chip.py`,
      `'tasksBannerNotices'` in `test_tab_tasks_offline_banner.py`, `'boundHeatmapAxes'` in
      `test_index_html.py::test_scheduler_heatmap_bounds_js_is_served`). ξ2 already edits those files and lands
      before ο, so it moves the four "the served module defines its global" checks into a node test and they are
      never listed;
    - the helper clause finds the two tests outside the 30 modules that import `_lock_chip_matrix.py`, which
      slices `cellStateFor` out of `scheduler_heatmap.jsx` by `extract_function_body`: `test_lock_chip_state.py`
      (5 tests) and `test_boundary_js.py`'s sketch-12 cell-and-chip test. π6 runs both through the jsdom
      window, where `window.DF_SCHED_HEATMAP.cellStateFor` and `window.DF_SCHED_UTILS.lockChipStateFor` are
      already exported, so π8 can delete `_lock_chip_matrix.py`.

    Owners on C6's list: each of the 30 modules is owned by its tab group's task (π1–π7, by the module lists in
    §Decomposition plan); the helper layer (`_dashboard_helpers.py`, `_app_jsx_source.py`, `_lock_chip_matrix.py`,
    `conftest.py`), `test_jsx_source_helpers.py`, `test_index_html.py` and `test_datum_access_paths.py` by π8;
    `test_charts_consumer_bindings.py` by π3; `test_prds_budget.py` and `test_tasks_budget.py` by π7;
    `test_lock_chip_state.py` and `test_boundary_js.py` by π6.

    The bundle-structure checks the sitting kept (OQ-3: `classic_script_scope.test.mjs`, `test_index_html.py`'s
    structure tests) are JavaScript or HTML checks and stay.
  - *Order.* A conversion task waits for the plan tasks that rewrite its tab or its test modules, so it never
    converts tests that are about to change: π1, π3, π4, π5 and π6 follow the stamp-removal tasks (ξ2, and ξ1
    through it); π6 follows κ (scheduler JSX), ζ (`tab_curator.jsx`) and ε (`tabs.jsx`); π4 follows γ1
    (`memory_readings.js`); π2 follows θ (`tabs.jsx`); every conversion follows 8461 (quality tab, in
    progress), and π3 also follows 8563. π8 follows π1–π7 and ο. Feature tasks that would otherwise add fresh
    source-text assertions to a module about to be rewritten (5816, 5922, 6254, 6926, 6947, 7931) wait on their
    tab's conversion task and are amended to test through the jsdom harness.
  - *Cost* (the sitting's measured estimate): about 8,400 new lines and about 17,000 deleted across the eight
    tasks; at 0.12 s per jsdom render test, about 400 more tests add roughly 50 s of node time before node's
    per-file parallelism; each task pays one full verify on landing.

## Contracts

```python
# C1 — dashboard/src/dashboard/project_labels.py (delivered by task 5760; S22 owns, no change)
@dataclass(frozen=True)
class ProjectLabels:
    @classmethod
    def of_roots(cls, roots: Iterable[Path]) -> ProjectLabels: ...   # labels by shortest unique suffix; never collides
    def including(self, roots: Iterable[Path]) -> ProjectLabels: ... # keeps every held label
    @property
    def roots(self) -> tuple[Path, ...]: ...
    def of(self, root: Path) -> str: ...            # unknown root -> basename_label(root)
    def root_for(self, label: str) -> Path | None: ...   # None = unknown; never guesses
    # direct construction (__post_init__) raises ValueError if one label would name two roots
```
Invariants: a bijection over the configured roots; a dashboard route that writes or drills down by label resolves
through `root_for` and refuses on `None`. Consumers: every dashboard reader and route keyed by project label.

```python
# C2a — dashboard/src/dashboard/data/mcp_fanout.py (γ1); egress names come from S19 C8 (dashboard/data/mcp_client.py)
FANOUT_FAILURE_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.TransportError, httpx.HTTPStatusError, TimeoutError, ValueError, OSError)
class FanoutFailureKind(StrEnum): DEADLINE = 'deadline'; UNREACHABLE = 'unreachable'; BAD_REPLY = 'bad_reply'
def classify_fanout_failure(exc: BaseException) -> FanoutFailureKind: ...   # TypeError for a non-member (a bug)
# first_success / call_with_deadline: signatures unchanged; catch FANOUT_FAILURE_EXCEPTIONS only;
# a failed URL's session is dropped through mcp_client.invalidate_session (S19 C8)

# C2b — dashboard/src/dashboard/data/escalation_probe.py (γ2)
@dataclass(frozen=True)
class ProbeReading(Generic[T]):
    label: str
    value: T | None
    failure: FanoutFailureKind | None
    detail: str | None                    # describe_exc(exc) when failure is set
async def probe_escalation_urls(
    urls: Mapping[str, str],              # label -> escalation endpoint URL (config.escalation_urls)
    tool: str, arguments: Mapping[str, object],
    project: Callable[[object], T],       # raises ValueError on a reply it cannot read -> BAD_REPLY
    *, deadline: float,
) -> dict[str, ProbeReading[T]]: ...
```
Invariants: exactly one reading per label. A label's failure never raises out of the helper, and only
`CancelledError` propagates. Base URLs come from `mcp_server_base`. A WARNING is logged when a label's state
changes, at DEBUG when it repeats, and at INFO on recovery. Consumers: merge_halt, merge_queue live queues,
escalation pins/recovery and task_runtime, which become `project` functions.

```python
# C3 — dashboard/src/dashboard/data/ttl_cache.py (δ1; CacheRegistry δ2)
DEFAULT_LOCK_ACQUIRE_TIMEOUT_SECONDS: float      # moved from mcp_fanout._LOCK_ACQUIRE_TIMEOUT_SECONDS; burndown imports it
@dataclass(frozen=True)
class CachePolicy:
    ttl_seconds: float                     # δ1 also accepts Callable[[], float]; δ2 removes the callable form
    lock_acquire_timeout_seconds: float = DEFAULT_LOCK_ACQUIRE_TIMEOUT_SECONDS
    max_live_bypasses_per_key: int = 3
    rewarn_every: int = DEFAULT_REWARN_EVERY
    eviction_ttl_multiple: int = 4
    # __post_init__: every field > 0, else ValueError naming the field and value
class SlotPhase(StrEnum): COLD; FRESH; STALE; REFRESHING; BYPASSING; AT_CAP
@dataclass(frozen=True)
class KeyState:
    phase: SlotPhase; age_seconds: float | None; waiters: int; live_bypasses: int; bypass_streak: int
class TTLCache(Generic[V, K]):
    def __init__(self, policy: CachePolicy): ...
    def get_fresh(self, key: K) -> V | None: ...
    async def get_or_refresh(self, key: K, refresh: Callable[[], Awaitable[V]], *,
                             cache_ok: Callable[[V], bool] = ...) -> V: ...
    def key_state(self, key: K) -> KeyState: ...
    async def cancel_live_bypasses(self) -> int: ...
    def clear(self) -> None: ...
async def reap_detached_refreshes() -> int: ...

# δ2 — the app-scoped registry (field names tactical; one field per module cache and ServedRead)
@dataclass(frozen=True)
class CacheRegistry:
    scheduler: TTLCache[...]; scheduler_state: ServedRead[...]; task_snapshot: TTLCache[...]
    task_lookup: TTLCache[...]; escalation_corpus: TTLCache[...]; served_corpus: ServedRead[...]
    analytics: TTLCache[...]; memory_evals: TTLCache[...]; prd_documents: ServedRead[...]; task_activity: ServedRead[...]
    @classmethod
    def build(cls, overrides: Mapping[str, CachePolicy] = {}) -> CacheRegistry: ...   # production policies + overrides
    def all(self) -> tuple[TTLCache[Any, Any], ...]: ...                              # reaper, cancel, clear
def install_caches(registry: CacheRegistry) -> None: ...   # lifespan installs app.state.caches; a test fixture installs its own
def caches() -> CacheRegistry: ...                          # the one accessor; LookupError before install
```
Invariants:
- One `_Slot` per key holds the value and its stamp, the lock, the waiter count, the tracked bypass, the live
  roster and the streak.
- A slot is removed only by `_reclaim(key)`, which every path calls: the done-callback, eviction, cancel and
  clear.
- No private attribute of an asyncio object is read.
- Lock waits are bounded by the policy, and a key has at most `max_live_bypasses_per_key` live bypasses.
- The reaper cancels only tasks on the running loop.
- (δ2) No module holds a `TTLCache` or `ServedRead` at import; readers reach theirs through `caches()`; tests
  install `CacheRegistry.build({...: CachePolicy(...)})` and never `setattr` a TTL constant.

Consumers: the 7 direct instantiation sites, `ServedRead`, `task_lineage`, and app lifespan's reaper call.

```python
# C4 — shared/src/shared/run_events.py and shared/src/shared/merge_outcomes.py (α)
class EventType(StrEnum): ...                       # moved verbatim from orchestrator/event_store.py (130 members at 1b106418da)
class OutcomeKind(StrEnum): ...                     # moved verbatim from orchestrator/merge_lane/types.py (23 members)
    @property
    def is_terminal(self) -> bool: ...
NON_TERMINAL_OUTCOMES: frozenset[OutcomeKind]

# dashboard/src/dashboard/data/outcome_colors.py (θ) — the consumer's family table, keyed by imported members
class OutcomeFamily(StrEnum): ...                   # the family set outcome_colors.py colours by (names tactical)
OUTCOME_FAMILY: Mapping[OutcomeKind, OutcomeFamily] # total over OutcomeKind; dashboard/tests/test_outcome_colors.py names a missing member
def family_of(recorded: str) -> OutcomeFamily | None: ...   # None: a recorded string that is not a member
```
Invariants: `orchestrator.event_store.EventType is shared.run_events.EventType`, and likewise for
`OutcomeKind`. Members are added in shared only. Nothing outside these two modules spells a member value,
whether as a class body, a mirror enum or an SQL literal. Consumers: `shared.runs_read`, dashboard,
fused-memory's `get_scheduler_events` callers, the client feed (its `StoreEventType` mirror goes) and the
scripts.

```python
# C5 — shared/src/shared/runs_read/ (β; grows by the migrating leaves, one query module per family)
@dataclass(frozen=True) class Read(Generic[T]): value: T
@dataclass(frozen=True) class Absent: path: Path
@dataclass(frozen=True) class Unreadable: path: Path; reason: str; error_type: str
ReadResult = Read[T] | Absent | Unreadable
@dataclass(frozen=True)
class RunsQuery(Generic[T]):
    sql: str; params: tuple[object, ...]; decode: Callable[[Sequence[sqlite3.Row]], T]   # decode is pure
    requires_columns: Mapping[str, frozenset[str]] = {}    # D4b
    fallback: RunsQuery[T] | None = None                   # run instead when a required column is missing
def open_runs_db(path: Path) -> sqlite3.Connection: ...   # tasks_db_readonly.connect_ro(path, store=SqliteStore.RUNS)
def read_runs_db(path: Path, query: RunsQuery[T]) -> ReadResult[T]: ...          # sync; one open per call
def run_query(conn: sqlite3.Connection, query: RunsQuery[T]) -> T: ...           # on a caller's snapshot; raises
async def read_runs_db_async(conn: aiosqlite.Connection | None, query: RunsQuery[T], *, path: Path) -> ReadResult[T]: ...
    # conn None: aiosqlite.connect(tasks_db_readonly.read_only_uri(path), uri=True)
# events.py (β; θ, υ add)
@dataclass(frozen=True)
class EventRow: id: int; timestamp: datetime; run_id: str | None; task_id: str | None; event_type: EventType | str; data: Mapping[str, object]
def events_of_types(types: Collection[EventType], *, since: datetime | None = None, until: datetime | None = None,
                    task_ids: Collection[str] | None = None, newest_first: bool = False, limit: int | None = None
                    ) -> RunsQuery[list[EventRow]]: ...
# task_results.py (β; η adds)
COUNTED_OUTCOMES: frozenset[str]        # the counted reading of shared.task_transitions.WORKFLOW_OUTCOMES (moved from performance)
@dataclass(frozen=True) class TaskResultRow: ...
def task_results_in_window(*, since: datetime, until: datetime | None = None, project_id: str | None = None,
                           counted_only: bool = True) -> RunsQuery[list[TaskResultRow]]: ...
# merge_lane.py (θ; σ adds)   audit.py (τ)   model_role.py (ι; υ uses)   quality.py, flake.py (ρ1)   scheduler.py (ρ2)
# markers.py (ρ3): the quality markers' routing_decision and config_reload event reads, runs.code_sha, invocations.model_id
# fixture.py — write_runs_db_fixture(path, *, events=(), task_results=(), runs=(), invocations=(), account_events=()):
#   the read model's fixture writer for any package's tests, checked against the real schema by the orchestrator test
# shared/src/shared/cost_store.py (ι): AccountEventType StrEnum (the account_events vocabulary usage_gate writes)
#   and RunsQuery-shaped invocations/account_events queries, used for runs.db and curator_events.db alike
```
Invariants:
- Outside `orchestrator/src`, no non-test module holds SQL text naming `events`, `task_results`, `invocations`,
  `account_events` or `flake_occurrence` except `shared/runs_read/` and `shared/cost_store.py`; and no non-test
  module outside them pairs any of runs.db's 12 tables in SQL text with the runs.db locator (`runs_db(`,
  `.runs_db`, `RUNS_DB` or the literal `runs.db`). A table match is
  `(?i)\b(FROM|JOIN)\s+(<tables>)\b(?![.\w])`, so a table name followed by `.` (as in "read from runs.db") is a
  path, not a table. ω's guard enforces both, with no allow-list entries.
- Queries hold no connection state.
- The opener never creates a file.
- `read_runs_db` and `read_runs_db_async` never raise on `sqlite3.Error`, `OSError` or a `TaskDbUnreadable`
  refusal; `run_query` raises, and the caller's opener owns the three-state mapping; a decode error propagates
  from all three.
- A query naming a column an older store may lack declares it in `requires_columns`; the executor answers
  `Unreadable(reason='schema older than <table>.<column>')` or runs the `fallback`.
- Every `invocations` query states its class under S20 C7: row-as-invocation (filters `returned = 1`) or
  sum-only (sums every attempt row).
- Every exported query has a row in the orchestrator contract test, which enumerates `shared.runs_read`'s exported
  query constructors (a new export without a row fails it) and runs each against a DB the real stores wrote and
  against the schema floor. It also checks that the fixture writer's tables and columns are a subset of
  `PRAGMA table_info` on a real store's file. Each leaf that adds a query module adds its rows; the test file is
  left to the scheduler's file lock (R54).
- The sync executors run only off the event loop (INV-8): through `asyncio.to_thread` or an executor, as
  `quality_series.py` and `quality_sampler.py` already call them; other dashboard and fused-memory reads use the
  async one.

Further queries (aggregations, payload projections, `runs`/`scheduler_state`/`flake_occurrence` reads) are added
by the leaf that migrates their first consumer, in that family's module and in this shape: `scheduler_state` by
ρ2 (task 8559's shared reader), `runs` by τ (`audit_model_admission.py`) and by ρ3 (the quality markers task 8460
landed in `dashboard/data/quality_markers.py`).

```python
# C6 — dashboard/tests/source_text_exemptions.py (ο creates; π1–π8 shrink it to empty) — names tactical
EXEMPT: Mapping[str, TaskRef]          # dashboard/tests module path -> the live conversion task that removes it
SOURCE_TEXT_EXEMPTIONS = governed_exceptions(          # shared.governed_exceptions: every key Debt(EXEMPT[key])
    'dashboard.tests.source_text_exemptions.EXEMPT', RULE, EXEMPT,
    dispositions={path: Debt(task) for path, task in EXEMPT.items()})
# dashboard/tests/test_no_frontend_source_assertions.py (ο): measures the offenders (C6 rule below) as a
# shared.ratchet.Enumeration and compares it with EXEMPT's keys: excess(...) fails naming the new module,
# slack(...) fails naming the converted module still listed
```
Invariants:
- An offender is a Python module under `dashboard/tests` that names a source-text helper (the helper-layer names
  in `_dashboard_helpers.py`, `_app_jsx_source.py`, `_lock_chip_matrix.py` and conftest's `app_jsx_body`) or
  reads the text of a `static/redux` `.js`/`.jsx` file (a `read_text`/`open` of a path under the redux
  directory, or `.text` of a served `/static/redux/*.js(x)` response). A test that only checks an asset's status
  code, and the JavaScript tests under `dashboard/tests/js/`, are not offenders. The guard module and its seeded
  fixture are excluded by path.
- Every exempted module is owned by a live conversion task (INV-12 owned debt); no entry is `Policy`, so no
  ratification row exists for this list.
- The list only falls: a module not on it fails the guard, and so does a listed module that no longer offends.
- After π8 the list is empty and the helper clause is gone with the helpers; the direct-read clause stays.

Consumers: π1–π8 (each deletes its own entries), and the next agent who writes a dashboard test.

**Upstream — S15 C4 (S15 owns; ε and ζ consume).** `get_status(panels={...})` returns `PanelsReply`
(`shared/dashboard_panels.py`): `ReconPanel | PanelUnavailable`, `MemoryOpsPanel | PanelUnavailable`,
`CuratorPanel | PanelUnavailable`. ε and ζ parse it into those models, map `PanelUnavailable` and a failed call to
UNKNOWN, and spell no fused-memory table or index. S15's `BurstStateRow` carries `effective_state`, computed
by the owner (`EventBuffer.burst_state_rows(*, now)` applies the instance's `burst_cooldown_seconds`), so ε
renders it and deletes `data/reconciliation.py::_DEFAULT_BURST_COOLDOWN` and the reader's cooldown comparison;
and its `PanelUnavailable` names the store and the exception class. S15's draft pins the 11 dashboard SQL sites
(`journal_entries` went with task 5890), so no point goes back to S15.

**Upstream — S19 C8 (S19 owns; γ1, γ2, δ1 consume).** `dashboard/data/mcp_client.py`: `mcp_tool_call`,
`invalidate_session`, `reset_sessions`, `enclosing_deadline`. S22 asks S19 κ2 to drop
`mcp_fanout.LOCK_ACQUIRE_TIMEOUT_SECONDS` from its public-name list, because δ1 moves the constant to
`ttl_cache.py::DEFAULT_LOCK_ACQUIRE_TIMEOUT_SECONDS`.

**Upstream — S20 C7 (S20 owns; ι, ρ1, ρ3, τ, υ consume).** `invocations` rows per attempt (`invocation_id`,
`attempt`, `returned`); a count reader filters `returned = 1`, a sum reader sums every row.

## Boundary-test sketch

| # | Side | Scenario | Pre | Post |
|---|---|---|---|---|
| 1 | C1 consumer | performance and costs for two runs.db under `/a/proj` and `/b/proj` | two roots, one project_id each | the panels are keyed by the two distinct `ProjectLabels`; neither overwrites the other |
| 2 | C2b | one orchestrator raises `httpx.ReadError` mid-response | three labels, one `MockTransport` failing | `/api/v2/dashboard/merge-queue` returns 200; that label's halt/live readings are unknown with failure `unreachable`; the other two are populated |
| 3 | C2b | a label replies with a non-dict | — | `bad_reply` for that label only; one WARNING on the transition, DEBUG on the second tick, INFO on recovery |
| 4 | C2a | URL 1 fails, URL 2 answers `get_queue_stats` | two fused-memory URLs | the reading comes from URL 2 (failover, not a sum); URL 1's session is invalidated through S19 C8; `classify_fanout_failure` names the kind |
| 5 | C3 | a refresh wedges past `lock_acquire_timeout_seconds` | `CachePolicy(ttl_seconds=1, lock_acquire_timeout_seconds=0.05, max_live_bypasses_per_key=2)` | caller 2 bypasses (`key_state` `BYPASSING`); caller 4 joins the newest live bypass (`AT_CAP`); on completion the slot is `FRESH` and `waiters == 0`; `reap_detached_refreshes` cancels the wedged task; no test touches a private |
| 6 | C3 | `CachePolicy(ttl_seconds=0)` | — | `ValueError` naming `ttl_seconds` and `0` |
| 7 | C4 producer | the orchestrator emits through the shared enum | real `EventStore` on `tmp_path` | `emit(EventType.merge_attempt, …)` round-trips; the member list equals a snapshot taken before α (moved, not changed) |
| 8 | C4 consumer (θ) | an `OutcomeKind` member has no colour family | a seeded family table missing one member, checked by the same totality check `test_outcome_colors.py` runs over the imported enum | the check fails naming the member; a retired recorded string renders "retired" |
| 9 | C5 producer | every exported query over a DB the real stores wrote | `EventStore`, `RunStore`, `CostStore`, flake ledger on `tmp_path` | typed rows with `EventType` members; fixture-writer DDL ⊆ the real `PRAGMA table_info`; an exported constructor with no row fails the test |
| 10 | C5 consumer | path absent / file locked or corrupt / decode bug | — | `Absent(path)` / `Unreadable(path, reason, error_type)` / the exception propagates |
| 11 | panels | fused-memory stopped | live dashboard against a stopped MCP | RECON_STATE, MEMORY_OPS and CURATOR_STATE carry UNKNOWN with the reason; the tabs render "—"; the next metrics tick writes NULL into `curator_snapshots.pending_total` and `capped_now` and into the memory/queue sampler columns |
| 12 | panels | one project's runs.db unreadable | chmod 000 on the fixture | `COSTS.summary.today` is null and renders "—"; cap intervals unknown, not "not capped"; other projects unchanged |
| 13 | performance | the same project in two runs.db | two fixture DBs | cards and history from the merged raw tallies equal the single-DB computation over the union; one DB raising makes that project UNKNOWN in cards AND history |
| 14 | scheduler | `collect_scheduler_state` result | two roots | a `SchedulerState` with named fields; each row has `task_id: int` and `project_root`; no consumer splits on `/T-` (a label containing `/T-` keeps its id) |
| 15 | discovery | a process whose `--config` project_root cannot be read | fake `ps` output | `RunningOrchestrator(root=Unresolved(...))`; the panel shows it unresolved; the metrics sampler and burndown both skip it through the one function |
| 16 | healthz | a `HealthBudget` whose per-probe timeouts exceed the total | — | `ValueError` at construction naming the values; tests build `HealthProbe` with a small budget; no `dashboard.app._*` patch |
| 17 | static | GET an asset twice | live app | `Cache-Control: no-cache` and an `ETag`; the conditional GET returns 304; after ξ2 `index.html` contains no `?v=` |
| 18 | guards | seeded violators | fixture modules | ω's encapsulation guard flags `FROM events` outside the read model, and flags a module naming `scheduler_state` in SQL beside `runs_db(`; a module holding `unknown_datum('… could not be read from runs.db')` beside `runs_db(` passes; a docstring and a `tests/` file are ignored; ο's guard (C6) flags a new test that imports a source-text helper and one that reads `static/redux/tabs.jsx` text, passes a test that only checks an asset's status code, and flags a listed module that no longer offends; a listed module whose owner is its conversion task passes |
| 19 | υ | the digest's model-role rollup and the dashboard's model-role cells | one fixture DB | identical rows from one shared query; `EventStore` drains before reading (S8 C5) |
| 20 | C5 quality (ρ1) | `rows_for` and `landings` over one fixture DB | a caller-held `sqlite3.Connection` | the same rows before and after ρ1, every query run on the caller's one snapshot through `run_query` |
| 21 | C5 schema skew | a project whose runs.db predates `task_results.charter_class` | the schema-floor fixture | every panel that does not need the column is served; a query requiring it without a fallback returns `Unreadable(reason='schema older than task_results.charter_class')`; `rows_for` uses its fallback |
| 22 | ρ2 | `get_scheduler_events` with an unknown event-type string, then with runs.db unreadable | fused-memory tool against a fixture | a ValidationError envelope naming the unknown string; a typed error envelope on `Unreadable`, an empty list only on `Absent`; the dashboard scheduler maps the error to UNKNOWN (κ) |
| 23 | π1–π8 | a converted tab's behaviours | the tab group's source-text modules and their list entries | each behaviour the deleted tests pinned and jsdom can see renders through `dom_harness.mjs` in the task's own `.test.mjs`; the modules and their C6 entries are gone; the commit message lists every dropped pin with its reason; after π8, `git grep -l -E 'strip_js_comments\|extract_function_body\|app_jsx_body\|_app_jsx_source\|jsx_prop_expr' -- dashboard/tests` is empty and the C6 list is empty |
| 24 | ρ3 | the quality markers over one fixture DB, then over the schema floor | a fixture runs.db with routing_decision, config_reload, runs and invocations rows | the same derived markers before and after ρ3, every read a `runs_read/markers.py` query run on the caller's connection; a store older than `runs.code_sha` yields no deploy markers, as today, through the query's `fallback` |

## Measured population — runs.db readers outside the orchestrator process

Measured at ba17934861, re-measured at 1b106418da where task 8460's landing changed it. Rule A (the locator):
`git grep -l -E 'runs\.db|runs_db|RUNS_DB' -- ':!orchestrator/src'
':!*.md' ':!*.jsonl'`. Rule B (SQL), case-insensitive over `*.py` and `*.sh`:
`(FROM|JOIN)\s+(events|task_results|runs|invocations|account_events|flake_occurrence|…)\b`. Every non-test hit
was checked by hand, excluding the same table names in reconciliation.db, link_heal.db and curator_events.db and
the prose hit in `fused-memory/.../reconciliation/context_assembler.py`'s docstring. runs.db's DDL owners
(`event_store.py`, `run_store.py`, `flake_ledger.py`, `shared/cost_store.py`) create 12 tables (rule:
`git grep -h -o -E 'CREATE TABLE( IF NOT EXISTS)? +[a-z_]+'` over those four).
Totals at 1b106418da: **29 direct reader files, 124 SQL sites (68 non-test sites on `events` outside
`orchestrator/src` and `plans/`, 73 with `plans/`), 6 of runs.db's 12 tables, 39 distinct `EventType` members
filtered** (ba17934861: 28 files, 120 sites, 66 and 71; task 8460 added `dashboard/data/quality_markers.py`'s
four sites, whose `routing_decision` and `config_reload` filters other readers already used). Drift (a
literal that is not a member): 0. Mirrors of the vocabulary: `client_feed/sources/events.py::StoreEventType`
(10 members), `verify_admission_cancel_on_red_census.py`'s `PLAN_EVENT_TYPE`, scheduler.py's plain-string recovery
types, and `model_role.py`'s hand copy of the digest SQL.

| Package | Direct reader files | SQL sites | Leaf |
|---|---|---|---|
| dashboard | 8 (`merge_queue`, `costs`, `cap_history`, `model_role`, `performance`, `task_activity`, `escalation_analytics`, `quality_markers`) + indirect (`quality_sampler.py::_measure_root`, `quality_series.py::_unsampled` and `quality_marker_sampler.py::_refresh_root`, which open runs.db and hand the connection on) | 46 | θ (events readers, escalation_analytics), η (performance), ι (costs, cap_history, model_role), ρ3 (quality_markers; its sampler opens through `open_runs_db`); the quality-series callers keep their connection and are unedited (ρ1) |
| fused-memory | 1 (`mcp_tools/scheduler_state.py::read_scheduler_events`) | 1 | ρ2 |
| client-feed | 1 (`sources/events.py`) + 2 indirect | 4 | ρ2 |
| shared | 3 (`quality_series/landings.py`, `quality_series/rows.py`, `cost_store.py`) + `quality_series/cli.py` (indirect: opens the connection) | 22 | ρ1 (quality_series); `cost_store.py` is an owner (ι adds its query values) |
| scripts | 14 + 3 indirect | 46 | σ (merge-lane family), τ (audit/operator family; `query-events.sh` deleted) |
| plans | 2 (`hardware-bench-2026-10-21/compare.py`, `summarize.py`) | 5 | out of scope: dated one-shot artefacts under `plans/`, outside the guard's scan |

In flight — tasks that add or reshape out-of-process runs.db reads (statuses read 2026-11-12 with `get_task`):

| Task | Module | Tables | Leaf |
|---|---|---|---|
| 8460 (quality-dashboard δ), **done** (`5a1f289e91`) | `dashboard/data/quality_markers.py`: `fold_routing_decisions` (`_ROUTING_SQL`), `read_config_changes` (`_CONFIG_RELOAD_SQL`), `deploy_markers`, `model_version_markers`; opened by `quality_marker_sampler.py::_refresh_root` | `events` (`routing_decision`, `config_reload`), `runs.code_sha`, `invocations.model_id` | landed before the amend could reach it; ρ3 moves the four reads into `runs_read/markers.py` (ρ3 → β, S20 ζ, 8465) |
| 8563 (instruments-honesty ε) | `scripts/sitting/fleet_state.py` (`pause_reason` from the `scheduler_state` row; `landed` onto `shared.quality_series.landings`) | `scheduler_state` | τ → 8563 moves the raw read onto ρ2's `scheduler_state` query; ω → 8563 |
| 8559 (instruments-honesty β1) | `shared/scheduler_state.py::read_scheduler_state_checked` | `scheduler_state` | ρ2 → 8559 moves it onto `runs_read/scheduler.py` |
| 8462 (quality-dashboard ζ) | writer: `task_results.charter_class`, a `task_blocked` payload field in `event_store.py` | — | no reader; D4b covers the new column on older stores |

`scripts/query-events.sh` (17 sites, last touched 2026-04-03, referenced by no file) is deleted by τ;
`scripts/task_event_timeline.py` is the referenced operator tool. Two cross-package couplings bear on S8's
scope:

- `scripts/sitting/fleet_state.py` imports `orchestrator.digest.query_count_done_in_window` and
  `query_model_role_rollup`/`ModelRoleRow` (`::spend_and_cap_hits`). Task 8563 rewrites `fleet_state.landed` onto
  `landings()`, so the first import is gone when S8 ι2 9231 (which already depends on 8563) moves that function.
  τ removes the second through ι's model-role query (τ → ι, 8563).
- `dashboard/data/model_role.py` copies the digest's model-role SQL. ι moves it into the read model, and υ
  makes the digest use that same query.

## Cross-PRD relationship

"X → Y" means X depends on Y. A wave-2 or wave-1 task is named by its id.

| Other stream / task | Direction | Mechanism | Owner | Order |
|---|---|---|---|---|
| S14 κ 6722 | consumes | `project_layout` paths, `McpSession`, `McpErrorType` in `app.py`, `config.py`, `api/merge_queue.py`, `escalation_corpus.py`, `loops.py`, `project_dbs.py`, `memory.py`, `tasks.py` | S14 | γ1, γ2, δ1, μ, ε, λ → 6722 |
| S19 κ1, κ2 (unfiled, wave 3) | consumes | C8 `dashboard/data/mcp_client.py` (the egress, the cycle); κ2's public names (`api/window.parse_window`, `project_dbs`, `loops`, `data/orchestrator.resolve_project_root`/`read_project_root_from_config`) | S19 | γ1, γ2, δ1 → S19 κ1; γ1, γ2, δ1, μ, ν, λ → S19 κ2 (same non-test files: `mcp_fanout.py` and `metrics.py` for γ1, `merge_queue.py` and `api/merge_queue.py` for γ2, `app.py` for μ; S19's κ2 says the fan-out leaves and the app.py partition depend on it); xref: κ2 drops `mcp_fanout.LOCK_ACQUIRE_TIMEOUT_SECONDS` (δ1 moves it) |
| S15 κ (unfiled) | consumes | C4 `PanelsReply` via `get_status(panels=…)` | S15 | ε, ζ → S15 κ (cross-plan edge wired at decompose); S15 C4 already serves `effective_state`, names the store in `PanelUnavailable` and pins 11 sites, so ε deletes `_DEFAULT_BURST_COOLDOWN` |
| S20 ζ (unfiled, wave 3) | consumes | C7 per-attempt `invocations` rows; `shared/cost_store.py`, `data/costs.py`, `data/model_role.py`, `digest.py`, `quality_series/rows.py`, the two admission scripts | S20 | ι, ρ1, ρ3, τ, υ → S20 ζ; each C5 invocations query states its class |
| S16 α (unfiled) | same file; consumes S16 C8 | `dashboard/data/orchestrator.py` (the `max_concurrent_tasks` default); `shared/src/shared/config_defaults.py` (C8, created by S16 α); `orchestrator/src/orchestrator/config.py` | S16 owns C8; λ adds `DEFAULT_ESCALATION_PORT` to it | λ → S16 α (lead rulings S16/OQ-4 option 2 and S22/OQ-5: the dashboard imports the C8 constants and takes no orchestrator dependency); λ's one-line `EscalationConfig.port` default edit sits on S16's `config.py` chain, placed by the lead at decompose |
| S2 η (unfiled, wave 3) | same file; consumes S2 C1 | escalation-status comparisons in `escalation_analytics.py`, `performance.py`, `escalations.py`, `escalation_corpus.py` | S2 | S2 η → θ, η, γ2, δ2 (S22's structural edits first, S2's operand swaps rebase onto them; agreed with S2's fold) |
| S1 ε1 (unfiled, wave 3) | same file | `merge_lane/types.py`, end of that chain | S1 | none: ε1 follows α transitively through the chain |
| S17 δ2 (unfiled, wave 3) | same file | `event_store.py` (after 9231) | S17 | none beyond the chain; a new `EventType` member goes in `shared/run_events.py` once α lands (xref) |
| S6 α 9140 | precedes | first on `merge_lane/types.py` (R54) | S6 | α → 9140 |
| task 7612 (done, `3b1f243a9b`) | landed | made the script's copy the public `NON_TERMINAL_OUTCOMES`, pinned by `TestOutcomeKindFrozenContract`, which loads the script by path | — | none left; α turns the copy into an import and deletes the pin |
| S3 α 9152 | follows | `event_store.py` and `merge_lane/types.py`; adds `EventType.merge_deduplicated` | S3 | 9152 → α (lead edge) and the OQ-1 amend |
| S7 α1 9136 | follows | adds `EventType.merge_gate_unmeasured` | S7 | already after 9152 on `event_store.py`; the OQ-1 amend |
| S6 β1 9144 | follows | adds 8 terminal `OutcomeKind` members; `test_outcome_kind.py`; `scripts/merge_lane_throughput.py` | S6 | 9144 → α, 9144 → σ (lead edges; 9144 is already parked behind 8528, so they cost it nothing); the OQ-1 amend |
| S4 λ 9146 | follows | `scripts/suite_census_evidence.py` | S4 | 9146 → σ (lead edge; 9146 is already parked behind 8528 and 5040) |
| S3 γ 9162 | precedes | import and prose repoints in `escalations.py`, `performance.py`, `escalation_corpus.py`, `scripts/sitting/fleet_state.py` | S3 | γ2, δ1, η, τ → 9162 |
| S10 ο 9285 | follows | `dashboard/data/task_snapshot.py` status members (consumer S10 ω 9286, a dependency of gate 6698) | S10 | 9285 → δ1 (lead edge; 9285 already waits on 8528 through 9283 → 9209 → 9255, so the edge costs S10 ω nothing) |
| S8 ι2 9231 | precedes | typed `EventStore` reads (C5 of S8); depends on 8563, 9230 | S8 | υ → 9231 |
| S25 κ 6765, δ 6759, ζ 6761, η 6762, λ 6766 (wave 1, pending) | same file | the scripts σ and τ migrate | S25 | σ → 6765; τ → 6759, 6761, 6762, 6765, 6766 (older, smaller leaves first; the lead reverses if they are lock-starved) |
| quality-dashboard (`plans/quality-dashboard-prd.md`): 8460 (done), 8461 (in progress), 8462, 8465 | same file; adds runs.db readers | 8460 landed `quality_markers.py` with four raw runs.db reads (`5a1f289e91`); 8461 `index.html`, `test_index_html.py`, `test_cache_buster_freshness.py::_REQUIRED_ASSETS` and four source-text test modules (`test_charts_null_samples.py`, `test_tab_escalation_analytics.py`, `test_app_poll_scope.py`, `test_index_html.py`); 8462 `event_store.py` (`charter_class`); 8465 `loops.py`, `quality_markers.py` (an instrument marker kind); the quality access-path guard (decisions 14, 15, 22) | quality-dashboard | ρ3 moves 8460's reads onto the read model and follows 8465 on `quality_markers.py`; every conversion task (π1–π8) → 8461 (ruled, OQ-4); ξ1 takes no edge onto 8461 (they share only test files); 8462 rebases under α (no member added); 8465 takes no edge from ζ (its loop step and ζ's removal of the tickets pool open are disjoint lines of `loops.py`, and an edge would hold high ζ behind the quality chain) |
| instruments-honesty (`plans/instruments-honesty-prd.md`): 8558 (in progress), 8559, 8561, 8563, 8564 | same file | 8558 `data/tasks.py`, `active_tasks.py`, `index.html`, `jsx_render.test.mjs` per its description (its live `metadata.files` no longer lists a dashboard file; the edges onto it are kept because it is running); 8559 `data/scheduler.py`, fused-memory `scheduler_state.py`, `shared/scheduler_state.py`; 8561 `event_store.py`, `data/merge_halt.py`; 8563 `fleet_state.py`, `digest.py`, `burndown.py`, `tabs.jsx`; 8564 `cost_store.py`, `event_store.py`, `burndown.py` | instruments-honesty | ξ1 → 8558; γ1 → 8558; κ → 8558, 8559, 8563; δ1 → 8559, 8563; ρ2 → 8559; λ → 8563; ε → 8563; θ → 8563; τ → 8563; υ → 8563; ω → 8563; π3 → 8563; 8561 → γ2 and 8564 → δ1, ι (lead edges: 8561 is parked behind 8528 and 8564 waits on it through 8561, so the S22 leaf goes first); amend 8561: its `data/merge_halt.py` read goes through `escalation_probe.py::probe_escalation_urls` as a projector; 8561 and 8564 also take the OQ-1 amend |
| feature tasks 5816 (medium), 5922, 6254, 6926, 6947, 7931 (pending, low) | same file | each declares a source-text module a conversion task deletes, and would add fresh source-text assertions to it | — | 5816 → π3, π5; 5922 → π5; 6254 → π2, π4; 6926 → π2, π5; 6947 → π1, π2; 7931 → π3 (lead edges, ruled at the sitting, OQ-4) and an amend to each: test through the jsdom harness |
| source-text helper upkeep 4244, 4287, 4351, 4461 (deferred), 6562, 7004, 7971, 9037, 9046 (pending, low) | made obsolete | each maintains or consolidates the source-text helper layer that π8 deletes | — | cancelled by the lead with `x_acceptance_reason` naming the absorbing conversion task (§Existing tasks) |
| 7907, 8703, 7566 (pending, low) | subsumed | render-test follow-ups on modules π1 and π7 convert | — | cancelled into π1 (7907, 8703) and π7 (7566) |
| 6963 (pending, low) | same file | about 25 stale "no JSX render harness" remarks, 10 of them in modules π1–π7 delete | — | amended to the remainder; no edge |
| task 7838 (pending, low) | same file | raw `file:{path}?mode=ro` opens in σ's and τ's scripts | — | 7838 → σ, τ (lead edges) and amend (its remaining sites) |
| task 8661 (pending, low) | shares the URI builder | `shared/async_sqlite_base.py::connect_read_only` | — | 8661 → β (lead edge) and amend: build the URI with `tasks_db_readonly.read_only_uri` |
| S23 (complete) | consumes | `sampler.store.SCHEMA` in `load.py` | S23 | none (ι touches `load.py` for `with_db` only) |
| S8 ι 9230 | adjacent | `event_store.py` writer | S8 | none beyond the chain through 9152 |

## Decomposition plan

Overlay bands apply. Same-file leaves are serialized by the edges in each `Deps:` line; test files are left to
the scheduler's file locks (R54). An edge that rests only on a shared test file is not wired, inside this plan or
across plans (lead ruling after the wave-3 fold-verifiers): δ1 and γ2 share only `_dashboard_helpers.py`, ι and
θ only `orchestrator/tests/test_runs_read_contract.py`, and ξ1 and 8461 only test modules, so those edges are
dropped and the files are rebase-only; υ, which shares `runs_read/events.py` with θ, now takes its own edge
onto θ. An edge between two leaves that share no non-test file rests on a contract the later leaf consumes
(β, κ → α; γ2 → γ1; κ, τ → ρ2; τ → ι; ρ1, ρ2, ρ3, σ, τ, ι → β; ξ2 → ξ1 by D11; ο → ξ2 and π8 → π1–π7, ο by
D14), or is an ordering Leo ruled under OQ-4 (π1, π3, π4, π5, π6 → ξ2; π2 → θ; π4 → γ1; π6 → κ, ζ, ε; every
π → 8461; π3 → 8563; the six feature tasks' edges onto π1–π7), or is ω's gate. `high` leaves (γ2, δ1, ζ, θ, ξ2, ω) become dependencies of the trigger-chain
gate 6698 at decompose. Every leaf that closes or contributes to a key carries `x_finding_key`,
`x_finding_run = review-all-dark_factory-20261008` and `source = review-all`. No leaf declares
`merge_lane/worker.py`. Over this plan's `Deps:` lines, the lead edges and the live store's edges, only υ
(through 9231) waits on 8528 or 5040. Adding the sibling wave-3 drafts' `Deps:` lines (re-checked at
`0f21f5e457`; no cycle through any S22 leaf), λ also waits on both, through S16 α, which depends on 9146 and
9229; ω inherits that through λ. Gate 6698 already waits on 8528 and 5040 (through 9156), so ω's wait adds
nothing to the gate. A filed task parked behind 8528 or 5040 that
shares a file with an S22 leaf takes the edge onto the S22 leaf (8561, 8564, 9285, 9144, 9146). The eight
conversion tasks π1–π8 and the guard ο are medium and carry fk-1bdd95e804cb; none is a dependency of gate 6698
(ω does not wait on them), as the sitting's discussion of OQ-4 assumed.

**α — Move the run-event and merge-outcome vocabularies into shared** (priority medium, ~500 LOC)
- Files: `shared/src/shared/run_events.py` (new), `shared/src/shared/merge_outcomes.py` (new),
  `shared/tests/test_run_vocabulary.py` (new), `orchestrator/src/orchestrator/event_store.py`,
  `orchestrator/src/orchestrator/merge_lane/types.py`, `orchestrator/tests/test_outcome_kind.py`,
  `orchestrator/tests/merge_lane_ratchet_baseline.json`, `scripts/merge_lane_throughput.py`,
  `scripts/tests/test_merge_lane_throughput.py`, `docs/task-authoring.md`
- Deps: 9140 (task 7612, which the draft also named, landed as `3b1f243a9b`)
- Keys: contributes to fk-c927adb040e9 (θ closes).
- Work: C4; D2. `scripts/merge_lane_throughput.py::NON_TERMINAL_OUTCOMES` (made public by task 7612) becomes an
  import of `shared.merge_outcomes.NON_TERMINAL_OUTCOMES`, so `TestOutcomeKindFrozenContract`, which since 7612
  loads the script by path and compares its set with the orchestrator's, compares one definition with itself and
  is deleted, with the FROZEN CONTRACT paragraph in `merge_lane/types.py`. `merge_lane/types.py` is measured by the merge-lane ratchet, so the baseline is
  re-measured in the same commit (a fall). `docs/task-authoring.md` gains one line: "an orchestrator event type is
  declared in `shared/src/shared/run_events.py`; a merge outcome in `shared/src/shared/merge_outcomes.py`".
- Deletes: the two class bodies in the orchestrator modules; the script's copy; `TestOutcomeKindFrozenContract`
  and the FROZEN CONTRACT prose that names it.
- Signal: `python -c "import shared.run_events as s, orchestrator.event_store as o; assert o.EventType is s.EventType"`
  passes in the root venv; boundary row 7.
- G7: INV-5/INV-9 (one home).

**β — Shared runs.db read model: query values, three-state results, the shared read-only opener** (medium, ~1,400 LOC)
- Files: `shared/src/shared/runs_read/__init__.py` (new), `shared/src/shared/runs_read/query.py` (new),
  `shared/src/shared/runs_read/events.py` (new), `shared/src/shared/runs_read/task_results.py` (new),
  `shared/src/shared/runs_read/fixture.py` (new), `shared/src/shared/tasks_db_readonly.py`,
  `shared/tests/test_runs_read.py` (new), `shared/tests/test_tasks_db_readonly.py`,
  `orchestrator/tests/test_runs_read_contract.py` (new), `orchestrator/tests/fixtures/runs_db_schema_floor.sql` (new)
- Deps: α
- Keys: none. It is the R36 deliverable; θ (high) depends on it, a priority inversion recorded and not
  overridden (contract §4).
- Work: C5; D3; D4; D4b. `connect_ro` gains a `store` keyword and a remedy table keyed by store, and its URI
  builder becomes the public `read_only_uri`. `run_query` executes on a caller's connection. The schema floor
  is captured from the oldest live runs.db in the fleet. The counted-outcome reading moves here from
  `performance.py::_READING_BY_OUTCOME` (η deletes the original).
- Signal: boundary rows 9–10 and 21 green in CI; the contract test writes through the real stores.
- G7: INV-13 (the reader proves its producer); INV-11; INV-2 (`Unreadable` carries path, reason, type); INV-8.

**γ1 — One fan-out failure classification; memory's queue and WAL reads fail over** (medium, ~600 LOC)
- Files: `dashboard/src/dashboard/data/mcp_fanout.py`, `dashboard/src/dashboard/data/memory.py`,
  `dashboard/src/dashboard/data/metrics.py`, `dashboard/src/dashboard/data/tasks.py`,
  `dashboard/tests/test_mcp_fanout.py`, `dashboard/tests/test_memory.py`, `dashboard/tests/test_metrics_data.py`,
  `dashboard/src/dashboard/static/redux/memory_readings.js`, `dashboard/tests/js/memory_readings.test.mjs`
- Deps: 6722, 8558, S19 κ1, S19 κ2 (`mcp_fanout.py`, `metrics.py`)
- Keys: fk-f3a5a34831e9. Contributes to fk-70e1ee9218a7 (tasks.py prose).
- Work: C2a; D6; D7. `memory.py::get_queue_stats` and `::get_wal_status` move onto `first_success` (failover,
  not a sum), with the partial-outage wording to match. `metrics.fan_out_list_tickets._call` catches the one
  tuple. The tasks.py module docstring is cut to its contract.
- Deletes: two hand-rolled URL loops; metrics' three-step except chain.
- Subsumes: 6872, 8119 (cancelled with `x_acceptance_reason` naming γ1).
- Signal: boundary row 4; with one of two fused-memory URLs refused, the Memory tab's write queue reads from
  the other URL and the partial-outage line names the failed path.
- G7: INV-11.

**γ2 — One concurrent escalation-URL probe; the four probes become projectors** (high, ~900 LOC)
- Files: `dashboard/src/dashboard/data/escalation_probe.py` (new), `dashboard/src/dashboard/data/merge_halt.py`,
  `dashboard/src/dashboard/data/merge_queue.py`, `dashboard/src/dashboard/data/escalations.py`,
  `dashboard/src/dashboard/data/task_runtime.py`, `dashboard/src/dashboard/api/merge_queue.py`,
  `dashboard/tests/test_escalation_probe.py` (new), `dashboard/tests/test_merge_halt.py`,
  `dashboard/tests/test_task_runtime.py`, `dashboard/tests/test_merge_queue_data.py`,
  `dashboard/tests/test_escalations_data.py`, `dashboard/tests/_dashboard_helpers.py`
- Deps: γ1, 6722, 9162, S19 κ1, S19 κ2 (`merge_queue.py`, `api/merge_queue.py`)
- Keys: fk-ad49b80c5554.
- Work: C2b. `api_merge_queue` stops bare-gathering. The three `_PerPortHandler` mocks become one helper's
  fixture.
- Deletes: 4 `removesuffix('/mcp')` copies, 4 exception tuples, `task_runtime`'s private log-dedupe state.
- Subsumes: 7628, 7503.
- Signal: boundary rows 2–3; on the live dashboard, `systemctl --user restart orchestrator-reify` during a poll
  leaves the merge tab rendering every other project.
- G7: INV-11; INV-4 (transition-only WARNING).

**δ1 — TTLCache in its own module, one slot per key, policy at construction** (high, ~1,200 LOC; 9 of the 18
files change by an import or constructor line)
- Files: `dashboard/src/dashboard/data/ttl_cache.py` (new), `dashboard/src/dashboard/data/mcp_fanout.py`,
  `dashboard/src/dashboard/data/served_read.py`, `dashboard/src/dashboard/api/escalations.py`,
  `dashboard/src/dashboard/api/memory_evals.py`, `dashboard/src/dashboard/app.py`,
  `dashboard/src/dashboard/data/escalation_corpus.py`, `dashboard/src/dashboard/data/scheduler.py`,
  `dashboard/src/dashboard/data/task_lookup.py`, `dashboard/src/dashboard/data/task_snapshot.py`,
  `dashboard/src/dashboard/data/burndown.py`, `dashboard/src/dashboard/data/task_lineage.py`,
  `dashboard/tests/test_ttl_cache.py` (new), `dashboard/tests/test_mcp_fanout.py`,
  `dashboard/tests/test_app_lifespan_reap.py`, `dashboard/tests/test_http_pool.py`,
  `dashboard/tests/test_tasks_budget.py`, `dashboard/tests/test_task_lineage.py`,
  `dashboard/tests/_dashboard_helpers.py`
- Deps: γ1, ν, 9162, 8559, 8563, S19 κ1, S19 κ2
- Keys: fk-b5760d0dfe85, fk-c51423b7b7e5.
- Work: C3 invariants 1–5; D8's δ1 half. Construction sites pass a `CachePolicy` (the callable `ttl_seconds`
  stays). `task_lineage`'s hand-rolled single flight goes onto C3. mcp_fanout keeps transport only, and its
  inventories and stale counts go (D13). The lock timeout moves to
  `ttl_cache.py::DEFAULT_LOCK_ACQUIRE_TIMEOUT_SECONDS`, and `data/burndown.py` imports it from there. Tests that
  patched `_LOCK_ACQUIRE_TIMEOUT_SECONDS` construct a policy.
- Deletes: the five per-key structures, the four reclaimers, the `_waiters` poke, every test setattr of the lock
  timeout, and `task_lineage._store`, `_start_refresh` and `_live_refresh`.
- Signal: boundary rows 5–6; `wc -l dashboard/src/dashboard/data/mcp_fanout.py` under 700 with no TTLCache; the
  dashboard's 14 endpoints answer during a wedged-key soak.
- G7: heuristic 10 (one reclaim decision); tests stance; INV-8.

**δ2 — The dashboard's caches are app-scoped; tests construct cache policies** (medium, ~900 LOC; 24 files, 12 of
them a test whose TTL-constant patch becomes an installed policy)
- Files: `dashboard/src/dashboard/data/ttl_cache.py`, `dashboard/src/dashboard/app.py`,
  `dashboard/src/dashboard/api/escalations.py`, `dashboard/src/dashboard/api/memory_evals.py`,
  `dashboard/src/dashboard/data/escalation_corpus.py`, `dashboard/src/dashboard/data/scheduler.py`,
  `dashboard/src/dashboard/data/task_lookup.py`, `dashboard/src/dashboard/data/task_snapshot.py`,
  `dashboard/src/dashboard/data/served_read.py`, `dashboard/src/dashboard/data/prd_signal.py`,
  `dashboard/src/dashboard/data/task_activity.py`, `dashboard/tests/conftest.py`, `dashboard/tests/test_ttl_cache.py`,
  `dashboard/tests/_boundary_payloads.py`, `dashboard/tests/test_active_tasks.py`, `dashboard/tests/test_api_prds.py`,
  `dashboard/tests/test_app.py`, `dashboard/tests/test_burndown_data.py`, `dashboard/tests/test_escalation_corpus.py`,
  `dashboard/tests/test_prd_signal.py`, `dashboard/tests/test_scheduler_page.py`,
  `dashboard/tests/test_scheduler_state_access.py`, `dashboard/tests/test_task_activity.py`,
  `dashboard/tests/test_tasks_census_projection.py`, `dashboard/tests/test_task_snapshot.py`
- Deps: δ1, θ, ξ1
- Keys: contributes to fk-b5760d0dfe85 (tests stance).
- Work: C3's `CacheRegistry`, `install_caches`, `caches()`; D8's δ2 half. Lifespan builds and installs the
  registry; `dashboard/tests/conftest.py` installs a fresh one per test. `CachePolicy.ttl_seconds` becomes a
  float.
- Deletes: the module-level `TTLCache`/`ServedRead` singletons; the callable-ttl form; the module TTL constants
  tests patched; every test `setattr` of one.
- Signal: `git grep -l -E "setattr\([^)]*_TTL_SECONDS|_TTL_SECONDS', " -- dashboard/tests` is empty; the live
  dashboard's cached endpoints answer as before.
- G7: heuristic 5 (scope); tests stance.

**ε — Recon and memory-ops panels read fused-memory's read model; the dashboard's fused-memory SQL goes**
(medium, ~1,000 LOC)
- Files: `dashboard/src/dashboard/data/reconciliation.py`, `dashboard/src/dashboard/data/write_journal.py`,
  `dashboard/src/dashboard/data/metrics.py`, `dashboard/src/dashboard/api/recon.py`,
  `dashboard/src/dashboard/api/memory_graphs.py`, `dashboard/src/dashboard/api/health.py`,
  `dashboard/src/dashboard/config.py`, `dashboard/src/dashboard/loops.py`, `dashboard/src/dashboard/data/redux_api.py`,
  `dashboard/src/dashboard/static/redux/tabs.jsx`, `dashboard/tests/test_reconciliation.py`,
  `dashboard/tests/test_write_journal.py`, `dashboard/tests/_dashboard_helpers.py`,
  `dashboard/tests/_boundary_payloads.py`
- Deps: ζ, ν, θ, 6722, 8563, S15 κ
- Keys: fk-1c1d247caaa7.
- Work: S15 C4 consumed; D5. The recon spark sampler reads the same panel.
- Deletes: 10 SQL sites over 7 tables (`runs`, `watermarks`, `event_buffer`, `burst_state`, `judge_verdicts`,
  `write_ops`, `read_op_counts`); `INDEXED BY` and `_MemoryOpsQuery`'s missing-object retry;
  `_DEFAULT_BURST_COOLDOWN` and the reader's cooldown comparison (the panel serves `effective_state`);
  `config.reconciliation_db`, `write_journal_db` and `write_queue_db` (no reader left); healthz's two fused-memory
  DB targets; 7 hand-copied fused-memory DDLs (5 in `_dashboard_helpers.py`, `test_write_journal.py`,
  `_boundary_payloads.py`).
- Signal: boundary row 11 for recon and memory ops;
  `git grep -E '(FROM|JOIN)\s+(runs|watermarks|event_buffer|burst_state|judge_verdicts|write_ops|read_op_counts)\b' dashboard/`
  is empty.
- G7: INV-13; INV-11; INV-9.

**ζ — The curator panel reports a fused-memory outage as one; samplers write NULL** (high, ~950 LOC)
- Files: `dashboard/src/dashboard/data/metrics.py`, `dashboard/src/dashboard/api/curator.py`,
  `dashboard/src/dashboard/data/redux_api.py`, `dashboard/src/dashboard/static/redux/tab_curator.jsx`,
  `dashboard/src/dashboard/loops.py`, `dashboard/src/dashboard/config.py`,
  `dashboard/tests/test_metrics_curator.py`, `dashboard/tests/test_api_curator.py`,
  `dashboard/tests/_curator_sampler_fixture.py`, `dashboard/tests/test_metrics_data.py`,
  `dashboard/tests/js/jsx_render.test.mjs`
- Deps: γ1, ν, S15 κ
- Keys: fk-4afbeb684a22. Contributes to fk-1c1d247caaa7 (the `tickets` site and its DDL copy).
- Work: D5. `fan_out_list_tickets` returns per-root reachability. `CURATOR_STATE` carries `pending`,
  `pending_total` and `capped_now` as Datums. `_sample_curator` reads `CuratorPanel` and writes NULL for an
  unmeasured column. The memory and queue samplers write NULL for fused-memory's `-1` and for zero-filled
  disconnected counts.
- Subsumes: 7815.
- Deletes: `offline_result=lambda errs: (0, [])`, the `([], 0)` substitution, the `tickets` SQL site, the
  `tickets` DDL copy, `config.tickets_db`, the sampler's `tickets_db` parameter and its pool open in `loops.py`.
- Signal: boundary row 11 for the curator tab (renders "—", names fused-memory) and `curator_snapshots`
  (`SELECT pending_total FROM curator_snapshots ORDER BY ts DESC LIMIT 1` is NULL after a tick with
  fused-memory stopped).
- G7: INV-11; INV-13.

**η — Performance: additive per-DB tallies, one merge, one partial-tally rule** (medium, ~1,100 LOC)
- Files: `dashboard/src/dashboard/data/performance.py`, `dashboard/src/dashboard/data/redux_api.py`,
  `dashboard/src/dashboard/api/performance.py`, `dashboard/src/dashboard/data/stats_utils.py`,
  `shared/src/shared/runs_read/task_results.py`, `orchestrator/tests/test_runs_read_contract.py`,
  `dashboard/tests/test_performance.py`, `dashboard/tests/test_redux_api.py`
- Deps: β, ε, ν, 9162
- Keys: fk-a9e45ca0eb10.
- Work: each family's per-DB step reads through C5 (its queries in `runs_read/task_results.py`) and returns
  counts or raw samples only. One `merge_tallies` merges and owns the partial rule: a project any DB failed to
  tally is UNKNOWN in cards and in history. Every pct, rate and percentile is derived once, after the merge. Loop
  bins are one module constant shared by step and merge. Labels come from each DB's root via C1.
  `_HISTORY_CACHE` memoizes raw per-DB tallies (tactical T3).
- Deletes: `get_time_centiles`; `_merge_history`'s unweighted mean; the unreachable label-merge branch and the
  4 monkeypatches reaching it; `_drop_partial_tallies` as a cards-only rule; history's `except Exception` →
  DEBUG; `_READING_BY_OUTCOME` (moved to β); `shape_performance`'s `basename_label` keying.
- Signal: boundary rows 1 and 13; on the live dashboard, single-DB projects' performance tab values are unchanged
  against a pre-change capture, and a project with two runs.db shows history equal to the pooled computation.
- G7: heuristic 3 (one mechanism per axis); INV-11.

**θ — Dashboard event readers on the read model; outcome families keyed by the imported members** (high, ~1,100 LOC)
- Files: `dashboard/src/dashboard/data/merge_queue.py`, `dashboard/src/dashboard/data/task_activity.py`,
  `dashboard/src/dashboard/data/escalation_analytics.py`, `dashboard/src/dashboard/data/outcome_colors.py`,
  `dashboard/src/dashboard/static/redux/tabs.jsx`, `shared/src/shared/runs_read/merge_lane.py` (new),
  `shared/src/shared/runs_read/events.py`, `orchestrator/tests/test_runs_read_contract.py`,
  `dashboard/tests/test_merge_queue_data.py`, `dashboard/tests/_runs_db_fixture.py`,
  `dashboard/tests/test_task_activity.py`, `dashboard/tests/test_escalation_analytics.py`,
  `dashboard/tests/test_outcome_colors.py`, `dashboard/tests/test_datum_access_paths.py`,
  `dashboard/tests/_quality_fixture.py`, `dashboard/tests/_boundary_payloads.py`
- Deps: β, γ2, 8563 (8460, which the draft also named, landed as `5a1f289e91`)
- Keys: fk-af4ba730a48a, fk-c927adb040e9.
- Work: 15 SQL sites (merge_queue 9, task_activity 5, escalation_analytics 1) become C5 queries; the merge-lane
  aggregations go to `runs_read/merge_lane.py`, task and escalation event reads use `runs_read/events.py`.
  `outcome_colors` holds `OutcomeFamily`, `OUTCOME_FAMILY` keyed by the imported `OutcomeKind` members, and
  `family_of`; `test_outcome_colors.py` asserts totality over `OutcomeKind`, and a retired code renders "retired".
  The merge-event row serves a typed badge field, so `tabs.jsx` spells no event type. In
  `test_datum_access_paths.py`, the three `data/merge_queue.py` grants retire and `GUARDED_QUALITY_READS` gains the
  `runs_read/merge_lane.py` constructors that return per-landing quality sources (D3, two guards). Dashboard tests
  build runs.db through `write_runs_db_fixture`.
- Deletes: `_CODE_FAMILY`; every event-type and outcome literal in these modules and the `'train_merged'` badge
  test in `tabs.jsx`; `MERGE_EVENTS_SCHEMA`; `_runs_db_fixture.py`; the runs.db DDL copies in these test files;
  escalation_analytics' private `sqlite3.connect` and `except Exception`.
- Signal: the merge-queue doughnut colours `main_health_red` by its family on the live dashboard;
  `git grep -E "'(merge_attempt|train_merged|task_started)'" dashboard/src` is empty; boundary row 8.
- G7: INV-5; INV-9; heuristic 12.

**ι — Cost, cap, model-role and load readers typed; with_db deleted** (medium, ~1,100 LOC)
- Files: `dashboard/src/dashboard/data/costs.py`, `dashboard/src/dashboard/data/cap_history.py`,
  `dashboard/src/dashboard/data/model_role.py`, `dashboard/src/dashboard/data/load.py`,
  `dashboard/src/dashboard/data/db.py`, `dashboard/src/dashboard/data/redux_api.py`,
  `shared/src/shared/cost_store.py`, `shared/src/shared/runs_read/model_role.py` (new),
  `orchestrator/tests/test_runs_read_contract.py`, `shared/tests/test_cost_store.py`,
  `dashboard/tests/test_costs_data.py`, `dashboard/tests/test_cap_history.py`,
  `dashboard/tests/test_model_role_data.py`, `dashboard/tests/test_load_data.py`
- Deps: β, η, ε, S20 ζ
- Keys: fk-9979f1b771de.
- Work: D4; D5. `AccountEventType` and the invocations/account_events query values live in `cost_store`, each
  invocations query stating its S20 C7 class. model_role's digest-mirror SQL becomes one query in
  `runs_read/model_role.py` (υ points the digest at it). `shape_costs` serves null, not `0.0`. Load reads the
  sampler store through the same three-state adapter.
- Deletes: `with_db`; 14 cost SQL sites restated outside `cost_store`; the cost DDL copies in dashboard tests.
- Signal: boundary row 12; `git grep -n 'with_db' dashboard/src` is empty.
- G7: INV-11; INV-9.

**κ — Scheduler collector: fetch plus a pure shaper, a named record, raw task ids on rows** (medium, ~1,300 LOC)
- Files: `dashboard/src/dashboard/data/scheduler.py`, `dashboard/src/dashboard/data/active_tasks.py`,
  `dashboard/src/dashboard/api/scheduler.py`, `dashboard/src/dashboard/data/redux_api.py`,
  `dashboard/src/dashboard/static/redux/scheduler_utils.jsx`, `dashboard/src/dashboard/static/redux/tabs.jsx`,
  `dashboard/src/dashboard/static/redux/shell.jsx`, `dashboard/src/dashboard/static/redux/prd_lanes_view.js`,
  `dashboard/src/dashboard/static/redux/tab_scheduler.jsx`, `dashboard/tests/test_scheduler_page.py`,
  `dashboard/tests/test_active_tasks.py`, `dashboard/tests/js/jsx_render.test.mjs`
- Deps: δ2, ι, α, ρ2, 8558, 8559, 8563
- Keys: fk-6aee6c49fa2e, fk-d21f64525e52, fk-70e1ee9218a7.
- Work: one fetch per root returns raw snapshots and events. A pure `shape_project(...) -> SchedulerProjectView`
  does the rest. `collect_scheduler_state -> SchedulerState` is frozen, with named fields, and keeps offline
  and degraded apart. Rows gain `task_id: int` and `project_root`. Events are keyed by `(project, task_id)`.
  The recovery and skip event types are `EventType` members. `get_scheduler_events`' typed error envelope (ρ2)
  maps to UNKNOWN. The `_TASKS_TOTAL_BUDGET` block and scheduler's stale rationale are rehomed (D13).
- Subsumes: 6842.
- Deletes: the 8-tuple and its three positional restatements; `rsplit('/T-')` and the three JSX splits; the
  composite event key; the "does not exist yet" prose.
- Signal: boundary row 14; the scheduler tab's override drawer POSTs the row's own `project_root` on a
  same-basename fixture.
- G7: heuristics 4, 8, 12.

**λ — Orchestrator discovery and the orchestrator-yaml read, one function each** (medium, ~850 LOC)
- Files: `dashboard/src/dashboard/data/orchestrator.py`, `dashboard/src/dashboard/config.py`,
  `dashboard/src/dashboard/data/burndown.py`, `dashboard/src/dashboard/data/metrics.py`,
  `shared/src/shared/config_defaults.py`, `orchestrator/src/orchestrator/config.py`,
  `dashboard/tests/test_orchestrator.py`, `dashboard/tests/test_config.py`, `dashboard/tests/test_burndown_data.py`,
  `dashboard/tests/test_metrics_data.py`
- Deps: ε, δ1, 6722, 8563, S16 α, S19 κ2
- Keys: fk-d0a7ebc4a4f1, fk-9cf2a161c9c0.
- Work: D12. `_read_escalation_url`, `read_project_root_from_config` (public after S19 κ2) and
  `read_max_concurrent_tasks` become views of one parsed `OrchestratorYaml`. λ adds `DEFAULT_ESCALATION_PORT`
  (8100) to `shared/src/shared/config_defaults.py` (S16 C8) and makes
  `orchestrator/src/orchestrator/config.py::EscalationConfig.port` name it as its field default; a missing
  `escalation.port` takes that constant, and the concurrency default is C8's `DEFAULT_MAX_CONCURRENT_TASKS`. The
  dashboard imports both and restates neither (lead rulings S16/OQ-4 option 2 and S22/OQ-5).
- Subsumes: 6875.
- Deletes: the two hand-rolled discovery walks in burndown and metrics; the second env-expansion behaviour; the
  dead `TimeoutExpired` clause.
- Signal: boundary row 15; the ORCHESTRATORS panel and the running-count history agree on a fleet with an
  unresolvable `--config`.
- G7: INV-5; heuristic 12.

**μ — /healthz probe as a HealthProbe on app.state** (medium, ~700 LOC)
- Files: `dashboard/src/dashboard/api/health.py` (new), `dashboard/src/dashboard/app.py`,
  `dashboard/tests/test_healthz_deadline.py`, `dashboard/tests/test_healthz_data_plane.py`
- Deps: 6722, S19 κ2 (`app.py`; κ2 brings κ1's `app.py` repoint onto `mcp_client`)
- Keys: fk-166cacf84a18.
- Work: D9. Lifespan builds the probe; the route reads `request.app.state.health`.
- Deletes: the module globals, `global`, `_mcp_probe_state_clear`, the four independent budget constants, the
  verdict-string membership tests, and 11 healthz private patch targets.
- Signal: boundary row 16; `curl /healthz` reports the same verdicts on the live service.
- G7: heuristic 5; heuristic 10; tests stance. Live deferred task 4790 (data-plane coverage) is not subsumed.

**ν — Remaining app.py routes into api/ routers with typed bodies** (medium, ~1,200 LOC)
- Files: `dashboard/src/dashboard/app.py`, `dashboard/src/dashboard/api/scheduler.py` (new),
  `dashboard/src/dashboard/api/curator.py` (new), `dashboard/src/dashboard/api/costs.py` (new),
  `dashboard/src/dashboard/api/performance.py` (new), `dashboard/src/dashboard/api/recon.py` (new),
  `dashboard/src/dashboard/api/memory_graphs.py` (new), `dashboard/src/dashboard/api/load.py` (new),
  `dashboard/src/dashboard/api/memory_evals.py` (new), `dashboard/src/dashboard/api/write_proxy.py` (new),
  `dashboard/tests/test_app_module_boundaries.py`, `dashboard/tests/test_scheduler_page.py`
- Deps: μ, S19 κ2
- Keys: fk-1b8c06cd8b2f, fk-ce66d4da9b0f.
- Work: D10. `app.py` ends as construction, lifespan, router wiring and `index`. The routers import the names S19
  κ2 made public. This is a move, and every resulting file passes heuristic 13: the measured partition has no
  reach-back.
- Subsumes: 8700.
- Deletes: about 250 lines of hand validation; `_VALID_BOOST_TIERS`; app.py's 16 `@app` routes but `/`,
  `/api/health` and `/healthz` (router).
- Signal: POSTing an unknown field or a non-configured `project_root` to `/api/v2/dashboard/scheduler/override`
  returns 422 naming the field; `wc -l app.py` under 500.
- G7: heuristics 6, 12, 14.

**ξ1 — Static assets revalidate; the tests stop pinning a `?v=` floor** (medium, ~350 LOC)
- Files: `dashboard/src/dashboard/static_assets.py` (new), `dashboard/src/dashboard/app.py`,
  `dashboard/tests/test_static_revalidation.py` (new), `dashboard/tests/test_charts_axis_labels.py`,
  `dashboard/tests/test_esc_flow_diagram.py`, `dashboard/tests/test_scheduler_page.py`,
  `dashboard/tests/test_tab_escalation_analytics.py`, `dashboard/tests/test_tab_escalations.py`,
  `dashboard/tests/test_tab_memory_evals.py`, `dashboard/tests/test_tab_scheduler.py`,
  `dashboard/tests/test_tab_tasks_invariant_chip.py`, `dashboard/tests/test_tab_tasks_offline_banner.py`,
  `dashboard/tests/test_tab_tasks_status_counts.py`
- Deps: ν, δ1, 8558
- Keys: contributes to fk-94fc76e7a123 (ξ2 closes); priority inversion with ξ2 (high) recorded, not overridden.
- Work: D11's first half. `/static/redux/*` and `/` answer with `Cache-Control: no-cache` and an ETag. The ten
  tests that pin a `?v=` floor or a sibling-version agreement lose those assertions (the tag-presence and
  load-order assertions stay); `test_tab_tasks_status_counts.py`'s tag regex accepts an unstamped tag; and
  `test_tab_tasks_invariant_chip.py` stops importing `_cache_buster_helpers`.
- Deletes: the `cache_buster_not_reverted_below_*` and `cache_buster_floor` tests and the sibling-version checks.
- Signal: boundary row 17's first half; a conditional GET of `/static/redux/tabs.jsx` returns 304.
- G7: INV-9.

**ξ2 — Strip the `?v=` stamps; the cache-buster freshness test goes** (high, ~350 LOC)
- Files: `dashboard/src/dashboard/static/redux/index.html`, `dashboard/src/dashboard/static/redux/charts.jsx`,
  `dashboard/tests/test_index_html.py`, `dashboard/tests/test_cache_buster_freshness.py`,
  `dashboard/tests/_cache_buster_helpers.py`, `dashboard/tests/test_tab_tasks_deferral.py`,
  `dashboard/tests/test_tab_tasks_invariant_chip.py`, `dashboard/tests/test_tab_tasks_offline_banner.py`,
  `dashboard/tests/js/_served_bundle.mjs`, `dashboard/tests/js/served_module_globals.test.mjs` (new)
- Deps: ξ1
- Keys: fk-94fc76e7a123.
- Work: D11's second half. `test_index_html.py` asserts no `?v=` on any local tag. `_served_bundle.mjs`'s comments
  stop describing a stamp (its regexes already accept an unstamped tag). The four served-module body checks
  (`'deferralCellState'`, `'invariantChipFor'`, `'tasksBannerNotices'`, `'boundHeatmapAxes'` in a served
  response's `.text`; D14 *Scope*) move into `served_module_globals.test.mjs`, which loads each module as
  `classic_script_scope.test.mjs` does through `_served_bundle.mjs` and asserts it binds its global; each Python
  test keeps its status-200 check. ξ2 lands before ο, so the four modules never go on C6's list. The commit
  message records the narrowed pin (a route serving the wrong file is now caught by the status check and the
  node load, not by a substring of the body). The Mem0 norm with topic
  `dashboard-redux-cache-buster` (cited in `fused-memory/tests/fixtures/README.md`) is superseded through
  `update_memory`/`supersedes`, so agents stop being told to bump.
- Deletes: 50 `?v=` occurrences, `test_cache_buster_freshness.py`, `_cache_buster_helpers.py`, the bump rule,
  `charts.jsx`'s bump comment, the four served-body substring assertions.
- Signal: boundary row 17; after a merge that changes `tabs.jsx`, a browser reload shows the change with no
  version bump.
- G7: INV-9.

**ο — Guard: no new dashboard test reads frontend source text; each exempted module is owned by its conversion
task** (medium, ~350 LOC)
- Files: `dashboard/tests/test_no_frontend_source_assertions.py` (new), `dashboard/tests/source_text_exemptions.py`
  (new), `dashboard/tests/fixtures/source_text_assertion_seeded.py` (new), `review/briefing.yaml`
- Deps: ξ2
- Keys: contributes to fk-1bdd95e804cb (π8 closes).
- Work: C6; D14. At landing the guard measures the offenders, and the list holds exactly those, each
  `Debt(TaskRef(<id>))` of the conversion task D14 assigns it (the decompose seat writes π1–π8's task ids into
  this task's description). The guard compares the measured offenders with the list through
  `shared/src/shared/ratchet.py`: a new offender fails (excess) and so does a listed module that no longer
  offends (slack). Its failure message points at `dashboard/tests/js/dom_harness.mjs` and names the listed
  owner. ξ2's deleted files are already gone, so they are never listed. `review/briefing.yaml` gains the
  convention "a dashboard frontend test renders through the jsdom harness". No row is added to
  `docs/legibility/exception-ratifications.yaml`.
- Signal: row 18; the guard flags its seeded fixture and passes the tree.
- G7: INV-10; INV-12 (every entry `Debt(owner)`, its owner a live conversion task).

**π1 — Tasks-tab source-text tests become render tests: runtime, status counts, focus header, staleness banner,
layout memo, terminal window, prose, module locks** (medium, ~1,750 new and ~1,680 deleted lines; 84 tests)
- Files: `dashboard/tests/js/render_tasks_tab.test.mjs` (new), `dashboard/tests/test_tab_tasks_runtime.py`,
  `dashboard/tests/test_tab_tasks_status_counts.py`, `dashboard/tests/test_tab_tasks_focus_header.py`,
  `dashboard/tests/test_tab_staleness_indicator.py`, `dashboard/tests/test_tab_tasks_layout_memo.py`,
  `dashboard/tests/test_tab_tasks_terminal_window.py`, `dashboard/tests/test_tab_tasks_prose.py`,
  `dashboard/tests/test_tab_tasks_module_locks.py`, `dashboard/tests/source_text_exemptions.py`,
  `dashboard/tests/js/wire_fixtures.mjs`, `dashboard/tests/js/jsx_render.test.mjs`
- Deps: ξ2, 8461
- Keys: contributes to fk-1bdd95e804cb (π8 closes).
- Work: D14. The eight modules' behaviours render through `dom_harness.mjs` in the new file; the modules are
  deleted and, if ο has landed, their entries leave C6's list. The staleness banner is rendered per tab as task
  7907 specified (exactly one banner for a stale path the open tab polls, none for a tab that does not), and the
  module-locks sentinel test in `jsx_render.test.mjs` gains the clause task 8703 asked for (it relies on
  `ModuleLocksSection` reading `window.DF_SCHED_UTILS` at render time). The layout-memo pins become a render-count
  check where jsdom can see one. The commit message lists every dropped pin with its reason. Slightly over the
  band on new lines (the sitting's estimate, an upper bound at one render test per deleted test).
- Subsumes: 7907, 8703 (cancelled with `x_acceptance_reason` naming π1).
- Deletes: the eight modules (84 tests, 1,676 lines, measured at 1b106418da).
- Signal: boundary row 23; `node --test dashboard/tests/js/render_tasks_tab.test.mjs` passes and the Tasks tab on
  the live dashboard is unchanged.
- G7: heuristic 10 (behaviour, not source grep); tests stance; INV-12.

**π2 — Overview and Orchestrators tabs and the app-chrome census become render tests** (medium, ~1,700 new and
~1,090 deleted lines; 81 tests)
- Files: `dashboard/tests/js/render_overview_orchestrators.test.mjs` (new), `dashboard/tests/test_tab_overview.py`,
  `dashboard/tests/test_tab_orchestrators.py`, `dashboard/tests/test_app_chrome_census.py`,
  `dashboard/tests/source_text_exemptions.py`, `dashboard/tests/js/wire_fixtures.mjs`
- Deps: θ, 8461
- Keys: contributes to fk-1bdd95e804cb (π8 closes).
- Work: D14, as π1, for the three modules (θ rewrites `tabs.jsx`'s merge badge first). Slightly over the band
  on new lines, as π1 (the sitting's estimate, an upper bound).
- Deletes: the three modules (81 tests, 1,089 lines).
- Signal: boundary row 23; the new file passes; the Overview and Orchestrators tabs and the rail badges are
  unchanged on the live dashboard.
- G7: heuristic 10; tests stance; INV-12.

**π3 — Burndown tab and chart source-text tests become render tests: axis labels, null samples, census tones,
chart consumer bindings** (medium, ~1,550 new and ~3,090 deleted lines; 68 tests)
- Files: `dashboard/tests/js/render_burndown_charts.test.mjs` (new), `dashboard/tests/test_tab_burndown.py`,
  `dashboard/tests/test_charts_axis_labels.py`, `dashboard/tests/test_charts_null_samples.py`,
  `dashboard/tests/test_census_tone_bindings.py`, `dashboard/tests/test_charts_consumer_bindings.py`,
  `dashboard/tests/source_text_exemptions.py`, `dashboard/tests/js/wire_fixtures.mjs`
- Deps: ξ2, 8461, 8563
- Keys: contributes to fk-1bdd95e804cb (π8 closes).
- Work: D14, as π1, for the four helper-rule modules, plus `test_charts_consumer_bindings.py` (4 tests, 274
  lines), which reads `.jsx` text to find dead `window.DF_CHARTS` bindings: it becomes a node check over the
  vendored Babel's parse of each `.jsx`, beside `classic_script_scope.test.mjs`, or its pins are dropped with the
  reason. Slightly over the band on new lines.
- Subsumes: the upkeep tasks 4351 and 7971 (their modules go here; cancelled naming π3).
- Deletes: the five modules (68 tests, 3,085 lines).
- Signal: boundary row 23; the new file passes; the Burndown tab and every chart render unchanged on the live
  dashboard.
- G7: heuristic 10; tests stance; INV-12.

**π4 — Memory and Memory-evals tab source-text tests become render tests** (medium, ~650 new and ~3,400 deleted
lines; 30 tests)
- Files: `dashboard/tests/js/render_memory_tabs.test.mjs` (new), `dashboard/tests/test_tab_memory.py`,
  `dashboard/tests/test_tab_memory_evals.py`, `dashboard/tests/source_text_exemptions.py`,
  `dashboard/tests/js/wire_fixtures.mjs`
- Deps: ξ2, γ1, 8461
- Keys: contributes to fk-1bdd95e804cb (π8 closes).
- Work: D14, as π1, for the two modules (γ1 rewrites `memory_readings.js` first).
- Deletes: the two modules (30 tests, 3,402 lines).
- Signal: boundary row 23; the new file passes; the Memory and Memory-evals tabs are unchanged on the live
  dashboard.
- G7: heuristic 10; tests stance; INV-12.

**π5 — Escalations and Escalation-analytics tabs, the flow diagram and the persisted-state consumers become
render tests** (medium, ~1,150 new and ~2,840 deleted lines; 54 tests)
- Files: `dashboard/tests/js/render_escalation_tabs.test.mjs` (new), `dashboard/tests/test_tab_escalations.py`,
  `dashboard/tests/test_tab_escalation_analytics.py`, `dashboard/tests/test_esc_flow_diagram.py`,
  `dashboard/tests/test_persisted_state_consumers.py`, `dashboard/tests/source_text_exemptions.py`,
  `dashboard/tests/js/wire_fixtures.mjs`
- Deps: ξ2, 8461
- Keys: contributes to fk-1bdd95e804cb (π8 closes).
- Work: D14, as π1, for the four modules.
- Deletes: the four modules (54 tests, 2,840 lines).
- Signal: boundary row 23; the new file passes; both escalation tabs and the flow diagram are unchanged on the
  live dashboard.
- G7: heuristic 10; tests stance; INV-12.

**π6 — Scheduler, Recon, Curator and Merge tabs, the park label and the chip-list hooks become render tests**
(medium, ~1,400 new and ~2,050 deleted lines; 59 tests converted, 6 rewired)
- Files: `dashboard/tests/js/render_ops_tabs.test.mjs` (new), `dashboard/tests/test_tab_scheduler.py`,
  `dashboard/tests/test_tab_recon.py`, `dashboard/tests/test_tab_curator.py`, `dashboard/tests/test_tab_merge_queue.py`,
  `dashboard/tests/test_park_entry_label.py`, `dashboard/tests/test_chip_list_hooks.py`,
  `dashboard/tests/test_lock_chip_state.py`, `dashboard/tests/test_boundary_js.py`,
  `dashboard/src/dashboard/static/redux/scheduler_heatmap.jsx`, `dashboard/tests/source_text_exemptions.py`,
  `dashboard/tests/js/wire_fixtures.mjs`
- Deps: ξ2, κ, ζ, ε, 8461
- Keys: contributes to fk-1bdd95e804cb (π8 closes).
- Work: D14, as π1, for the six modules (κ rewrites the scheduler JSX, ζ `tab_curator.jsx`, ε `tabs.jsx`'s recon
  panel first). The two importers of `_lock_chip_matrix.py` outside the six (D14 *Scope*) stop slicing
  `cellStateFor` out of `scheduler_heatmap.jsx`: `test_lock_chip_state.py`'s matrix and `test_boundary_js.py`'s
  sketch-12 cell-and-chip test call `window.DF_SCHED_HEATMAP.cellStateFor` and
  `window.DF_SCHED_UTILS.lockChipStateFor` in the jsdom window from `buildDashboardWindow`, either from
  `render_ops_tabs.test.mjs` or through a node driver the Python tests call, so neither imports
  `_lock_chip_matrix.py`. `scheduler_heatmap.jsx`'s header line naming the test that executes `cellStateFor` is
  repointed (D13); no code line changes.
- Deletes: the six modules (59 tests, 1,898 lines); the two tests' `_lock_chip_matrix` imports.
- Signal: boundary row 23; the new file passes; the four tabs are unchanged on the live dashboard.
- G7: heuristic 10; tests stance; INV-12.

**π7 — App-shell source-text tests become render tests: Datum components, poll scope, tab error boundary, the
browser abort timeout** (medium, ~650 new and ~1,350 deleted lines; 25 tests converted, 2 rewired)
- Files: `dashboard/tests/js/render_app_shell.test.mjs` (new), `dashboard/tests/js/_abort_delay_probe.mjs` (new),
  `dashboard/tests/test_datum_components.py`, `dashboard/tests/test_app_poll_scope.py`,
  `dashboard/tests/test_app_tab_error_boundary.py`, `dashboard/tests/test_prds_budget.py`,
  `dashboard/tests/test_tasks_budget.py`, `dashboard/tests/source_text_exemptions.py`,
  `dashboard/tests/js/wire_fixtures.mjs`
- Deps: 8461
- Keys: contributes to fk-1bdd95e804cb (π8 closes).
- Work: D14, as π1, for the three modules. The tab error boundary is rendered as task 7566 asked: a child that
  calls the real `assertDatum` on a bare number is replaced by the tab-render-failure banner while the rail and
  the stale banner survive. The two `DEFAULT_TIMEOUT_MS` reads (`test_prds_budget.py::_browser_abort_ms` and
  `test_tasks_budget.py::_browser_abort_ms`) stop reading `data.js` text: `_abort_delay_probe.mjs`
  loads `data.js` through `dashboard/tests/js/_data_loader.mjs`, drives one fetch that never answers and prints
  the abort delay it armed, and both budget tests read that number through node; if that proves unworkable the
  pin is dropped with its reason. The rest of the two budget modules stays.
- Subsumes: 7566 (cancelled naming π7); the upkeep task 7004 (cancelled naming π7; π1 and π2 delete its other
  two modules).
- Deletes: the three modules (25 tests, 1,331 lines) and the two `_browser_abort_ms` source reads.
- Signal: boundary row 23; the new file passes; both budget tests read the delay through the probe.
- G7: heuristic 10; tests stance; INV-12.

**π8 — Delete the source-text helper layer; the last served-script reads become node tests; the exemption list
ends empty** (medium, ~300 new and ~1,900 deleted lines; 2 tests converted, 109 deleted)
- Files: `dashboard/tests/js/retired_globals.test.mjs` (new), `dashboard/tests/test_index_html.py`,
  `dashboard/tests/test_datum_access_paths.py`, `dashboard/tests/test_jsx_source_helpers.py`,
  `dashboard/tests/_app_jsx_source.py`, `dashboard/tests/_lock_chip_matrix.py`, `dashboard/tests/_dashboard_helpers.py`,
  `dashboard/tests/conftest.py`, `dashboard/tests/source_text_exemptions.py`,
  `dashboard/tests/test_no_frontend_source_assertions.py`, `dashboard/tests/fixtures/source_text_assertion_seeded.py`
- Deps: π1, π2, π3, π4, π5, π6, π7, ο
- Keys: fk-1bdd95e804cb (closes).
- Work: D14. `test_index_html.py`'s one served-script-text read (the `strip_js_comments` reader behind
  `test_orch_summary_js_is_retired`) and `test_datum_access_paths.py::test_no_served_script_binds_a_retired_client_name`
  become one node test over the served bundle, reading it as `classic_script_scope.test.mjs` does through
  `_served_bundle.mjs`; `test_index_html.py`'s structure tests and the access-path guard's Python checks stay
  (OQ-3). Then the JS source-text helpers in `_dashboard_helpers.py` go (`_scan_js` through `jsx_prop_expr`; the
  HTML script-tag helpers `test_index_html.py` uses stay), with `_app_jsx_source.py`, `_lock_chip_matrix.py`
  (whose last importers π6 rewired), conftest's nine served-script fixtures (`app_jsx_body` and the eight others
  D14 *Scope* names; `index_html_body` stays) and the helpers' own contract module `test_jsx_source_helpers.py`.
  The guard
  drops its helper clause and its helper-import seeded violator (the direct-read clause and violator stay), and
  the C6 list ends empty with its declaration in place. No file under `dashboard/tests` names a removed helper
  afterwards, prose included. The commit message lists every dropped pin with its reason, as π1–π7 do.
- Subsumes: the upkeep tasks 4244, 4287, 4461, 6562 and 9046 (cancelled naming π8) and 9037 (cancelled naming
  π3, π5 and π7, which delete its three readers' modules).
- Deletes: the helper layer (four files' worth), `test_jsx_source_helpers.py` (109 tests, 1,433 lines), two
  served-script source reads.
- Signal: boundary row 23; `git grep -l -E 'strip_js_comments|extract_function_body|app_jsx_body|_app_jsx_source|jsx_prop_expr' -- dashboard/tests`
  is empty and the C6 list is empty.
- G7: heuristic 10; INV-12 (the list is empty); tests stance.

**ρ1 — Quality series on the read model** (medium, ~600 LOC)
- Files: `shared/src/shared/quality_series/landings.py`, `shared/src/shared/quality_series/rows.py`,
  `shared/src/shared/quality_series/cli.py`, `shared/src/shared/runs_read/quality.py` (new),
  `shared/src/shared/runs_read/flake.py` (new), `orchestrator/tests/test_runs_read_contract.py`,
  `shared/tests/test_quality_series_landings.py`, `shared/tests/test_quality_series_rows.py`,
  `shared/tests/quality_series_fixtures.py`, `dashboard/tests/test_datum_access_paths.py`
- Deps: β, S20 ζ
- Keys: none (R36 deliverable).
- Work: `landings`, `recorded_landing_keys` and `rows_for` keep their `sqlite3.Connection` parameter and run
  `RunsQuery` values from `runs_read/quality.py` and `runs_read/flake.py` on it through `run_query`, so the one
  snapshot `rows_for` relies on is kept and the callers (`quality_sampler.py::_measure_root`,
  `quality_series.py::_unsampled`, after 8563 `fleet_state.py` and `digest.py`) are unedited.
  `_has_charter_column` becomes a `requires_columns` query with a fallback (D4b). The invocations query states its
  S20 C7 class. `cli.py` opens through `open_runs_db`, so its refusal speaks of runs.db.
  `GUARDED_QUALITY_READS` gains the `runs_read/quality.py` constructors (D3, two guards).
- Deletes: 20 SQL sites in `landings.py` and `rows.py`; `_has_charter_column`.
- Signal: boundary rows 20–21; the dashboard's quality-series tick answers unchanged against the live stores.
- G7: INV-13; INV-11.

**ρ2 — Client feed and fused-memory's scheduler reads on the read model** (medium, ~400 LOC)
- Files: `client-feed/src/client_feed/sources/events.py`, `client-feed/src/client_feed/sources/store.py`,
  `client-feed/tests/test_sources_events.py`, `fused-memory/src/fused_memory/mcp_tools/scheduler_state.py`,
  `fused-memory/tests/test_scheduler_state_tools.py`, `shared/src/shared/scheduler_state.py`,
  `shared/tests/test_scheduler_state.py`, `shared/src/shared/runs_read/scheduler.py` (new),
  `orchestrator/tests/test_runs_read_contract.py`
- Deps: β, 8559
- Keys: none (R36 deliverable).
- Work: the client feed reads through `events_of_types` and its `task_results` aggregate. `get_scheduler_events`
  reads through C5: an MCP caller's `event_types` strings are parsed into `EventType`, and an unknown string
  returns a ValidationError envelope naming it (a typo is never an empty list); `Unreadable` returns a typed error
  envelope, and an empty list means `Absent` only. Task 8559's `shared/scheduler_state.py::read_scheduler_state_checked`
  reads the `scheduler_state` row through `runs_read/scheduler.py`, which τ reuses.
- Deletes: `StoreEventType` (10-member mirror); the client feed's 4 SQL sites and its own read-only opener;
  `get_scheduler_events`' raw read; 8559's raw `scheduler_state` read.
- Signal: `client-feed dump-sources --config <policy>` (the `dump-sources` subcommand of
  `client_feed/cli.py`) prints the same event sources for a fixture runs.db as a pre-change capture;
  `get_scheduler_events` and `get_scheduler_state` answer unchanged against the live stores; boundary row 22.
- G7: INV-13; INV-11.

**ρ3 — Quality markers read runs.db through the read model** (medium, ~500 LOC)
- Files: `dashboard/src/dashboard/data/quality_markers.py`, `dashboard/src/dashboard/data/quality_marker_sampler.py`,
  `shared/src/shared/runs_read/markers.py` (new), `orchestrator/tests/test_runs_read_contract.py`,
  `dashboard/tests/test_quality_markers.py`, `dashboard/tests/test_quality_marker_sampler.py`
- Deps: β, S20 ζ, 8465
- Keys: none (R36 deliverable; the reads landed after the run, with task 8460, `5a1f289e91`).
- Work: C5. The four reads in `quality_markers.py` (`_ROUTING_SQL`, `_CONFIG_RELOAD_SQL`, and the `runs.code_sha`
  and `invocations.model_id` reads inside `deploy_markers` and `model_version_markers`) become `RunsQuery` values
  in `runs_read/markers.py`, filtering on `EventType` members (C4). `fold_routing_decisions`,
  `read_config_changes`, `deploy_markers` and `model_version_markers` keep their `sqlite3.Connection` parameter
  and run them through `run_query`, so the sampler's one connection per refresh is kept. The two `_columns`
  probes become `requires_columns` declarations with an empty `fallback`, so a store older than `runs.code_sha`
  or `invocations.model_id` yields no markers from that source, as today (D4b). The invocations query states
  its S20 C7 class. `quality_marker_sampler.py::_refresh_root` opens through `open_runs_db`, so a refusal names
  runs.db. It follows 8465, the older task on `quality_markers.py`.
- Deletes: four SQL texts and `_columns` in `quality_markers.py`.
- Signal: boundary row 24; the Quality tab's markers on the live dashboard are unchanged against a pre-change
  capture.
- G7: INV-13; INV-5.

**σ — Merge-lane scripts on the read model** (medium, ~800 LOC)
- Files: `scripts/merge_lane_throughput.py`, `scripts/merge_deep_canary_predicate.py`,
  `scripts/merge-pytest-n-ab-analysis.py`, `scripts/analyze_speculation_depth.py`,
  `scripts/replay_deep_stack_study.py`, `scripts/verify_admission_cancel_on_red_census.py`,
  `scripts/suite_census_evidence.py`, `shared/src/shared/runs_read/merge_lane.py`,
  `orchestrator/tests/test_runs_read_contract.py`, `scripts/tests/test_merge_lane_throughput.py`,
  `scripts/tests/test_merge_deep_canary_predicate.py`, `scripts/tests/test_analyze_speculation_depth.py`,
  `scripts/tests/test_replay_deep_stack_study.py`, `scripts/tests/test_verify_admission_cancel_on_red_census.py`
- Deps: β, θ, 6765
- Keys: none (R36 deliverable).
- Work: their own `_connect_ro` copies go; runs.db opens go through `open_runs_db`, and the queries they need join
  θ's `runs_read/merge_lane.py`. This takes over the runs.db opens task 7838 listed for these scripts.
- Deletes: the `PLAN_EVENT_TYPE` copy; 3 private openers; 14 SQL sites.
- Signal: `scripts/merge_lane_throughput.py --project-root /home/leo/src/dark-factory --window 14d --json`
  prints the same report as a pre-change capture.
- G7: INV-5.

**τ — Audit and operator scripts on the read model; query-events.sh deleted** (medium, ~900 LOC)
- Files: `scripts/audit_model_admission.py`, `scripts/review_model_admission.py`,
  `scripts/audit_wiped_metadata_files.py`, `scripts/census_tagger_debris.py`, `scripts/mint_hard_v2_fixtures.py`,
  `scripts/task_event_timeline.py`, `scripts/sitting/fleet_state.py`, `scripts/query-events.sh`,
  `shared/src/shared/runs_read/audit.py` (new), `orchestrator/tests/test_runs_read_contract.py`,
  `scripts/tests/test_audit_model_admission.py`, `scripts/tests/test_audit_wiped_metadata_files.py`,
  `scripts/tests/test_task_event_timeline.py`, `scripts/tests/test_sitting_fleet_state.py`
- Deps: β, ι, ρ2, 9162, 8563, 6759, 6761, 6762, 6765, 6766, S20 ζ
- Keys: none (R36 deliverable).
- Work: `fleet_state` stops importing `orchestrator.digest` (its model-role figures come from ι's
  `runs_read/model_role.py`; its `pause_reason`, which 8563 adds, from ρ2's `runs_read/scheduler.py`). The `runs`
  read in `audit_model_admission.py` becomes a query in `runs_read/audit.py`. Invocations queries state their S20
  C7 class. This takes over the runs.db opens task 7838 listed for these scripts.
- Deletes: `scripts/query-events.sh`; `audit_model_admission.connect_ro`; `census_tagger_debris._connect_readonly`;
  32 SQL sites.
- Signal: `scripts/task_event_timeline.py <id>` prints the same timeline for a fixture task as before.
- G7: INV-5.

**υ — EventStore's typed reads delegate their SQL to the read model; one model-role query** (medium, ~600 LOC)
- Files: `orchestrator/src/orchestrator/event_store.py`, `orchestrator/src/orchestrator/digest.py`,
  `shared/src/shared/runs_read/events.py`, `shared/src/shared/runs_read/model_role.py`,
  `orchestrator/tests/test_event_store.py`, `orchestrator/tests/test_runs_read_contract.py`
- Deps: 9231, β, θ (`runs_read/events.py`), ι, 8563, S20 ζ
- Keys: none.
- Work: each `EventStore` read drains (S8 C5), then runs a C5 query; `digest.model_role_rollup` and the
  dashboard's model-role cells call one query.
- Deletes: the SQL inside `event_store.py`'s typed reads.
- Signal: boundary row 19; `git grep -n -E '(FROM|JOIN)\s+events' -- orchestrator/src/orchestrator/event_store.py orchestrator/src/orchestrator/digest.py`
  is empty; S8's `test_event_store_encapsulation.py` allow-list is unchanged.
- G7: INV-5; R36's guard keeps passing.

**ω — Integration gate: the dashboard's readers end to end; the runs.db read encapsulation guard** (high,
~650 LOC)
- Files: `dashboard/tests/test_readers_boundary.py` (new), `shared/tests/test_runs_db_read_encapsulation.py` (new),
  `shared/tests/fixtures/raw_runs_read_seeded.py` (new)
- Deps: α, β, γ1, γ2, δ1, δ2, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ1, ξ2, ο, ρ1, ρ2, ρ3, σ, τ, 8563 (not π1–π8: the
  medium conversion tasks stay off gate 6698's path; 8460, which the draft also named, is done)
- Keys: none (signal leaf).
- Work: rows 1–18, 20–22 and 24 on real fixtures (19 is υ's; 23 is each conversion task's). The guard derives its roots from `pyproject.toml`
  `[tool.uv.workspace].members` plus `scripts/`, excludes `**/tests/**`, and matches case-insensitively. SQL
  text is a Python string constant that is not a docstring (AST), or a line of a `*.sh` file. It applies C5's two
  rules: C5's five-table set outside the owners, and the co-occurrence of any of the 12 runs.db tables
  with the runs.db locator. A table match is `(?i)\b(FROM|JOIN)\s+(<tables>)\b(?![.\w])`, so a table name followed
  by `.` (as in "could not be read from runs.db", an `unknown_datum` reason D5 keeps) is a path, not a table. The owners (`shared/src/shared/runs_read/`, `shared/src/shared/cost_store.py`) are
  derived from C5, so the guard has no entries.
- Signal: the guard flags its seeded module and passes the tree; the §Goal checks hold on the live dashboard.
- G7: INV-10; INV-12 (no list).

## Existing tasks this PRD subsumes or orders (contract §8 step 2; statuses read 2026-11-12)

Step 1 (`find_tasks_by_metadata(x_finding_key=…)`, re-run 2026-11-12) found no task for any of the 22 keys. Step 2
found these, each confirmed with `search_tasks` ≥ 0.5 and read with `get_task`. Cancels carry
`x_acceptance_reason` naming the leaf. Per gotcha (a), a cancel whose target is in progress becomes an edge; none
of the targets below is in progress.

- 6872 FANOUT_FAILURE_EXCEPTIONS lacks the TransportError family (subsumed by γ1)
- 8119 write-queue stats as failover (subsumed by γ1)
- 7628 merge_halt `mcp_server_base` (subsumed by γ2)
- 7503 hoist `_PerPortHandler` (subsumed by γ2)
- 6842 named scheduler record (subsumed by κ)
- 6875 burndown/metrics discovery walks (subsumed by λ)
- 8700 app.py route extraction (subsumed by ν)
- 7815 samplers persist unmeasured values (subsumed by ζ)
- 7907 staleness-banner render test, 8703 module-locks sentinel comment (subsumed by π1); 7566 tab error boundary
  behaviour (subsumed by π7) — ruled at the sitting (OQ-4)

Made obsolete by the conversion (ruled at the sitting, OQ-4; cancelled by the lead, each naming its absorbing task):

- 4244 (deferred) consolidate the function-body extractor copies → π8
- 4287 (deferred) hoist the comment-stripping fixture → π8
- 4351 (deferred) dedupe the `window.DF_CHARTS` destructure parser → π3
- 4461 (deferred) consolidate the remaining copied source-assertion helpers → π8
- 6562 (pending, low) split the source-text helpers into their own module → π8
- 7004 (pending, low) move the app.jsx structural tests onto `_app_jsx_source.py` → π7 (π1 and π2 delete its
  other two modules)
- 7971 (pending, low) make a `DF_CHARTS` regex miss fail loudly in `test_tab_burndown.py` → π3
- 9037 (pending, low) move ad-hoc JSX tag readers onto `jsx_open_tags` → π3, π5 and π7 (they delete the three
  readers' modules)
- 9046 (pending, low) word-boundary bug in `jsx_prop_expr` → π8

Ordered, not subsumed:

- 5816 (pending, medium), 5922, 6254, 6926, 6947, 7931 (pending, low): each declares a source-text module a
  conversion task deletes; each takes an edge onto that tab's conversion task (5816 → π3, π5; 5922 → π5; 6254 →
  π2, π4; 6926 → π2, π5; 6947 → π1, π2; 7931 → π3) and an amend to test through the jsdom harness (ruled at the
  sitting, OQ-4).
- 6963 (pending, low) corrects about 25 stale "no JSX render harness" remarks; the 10 inside modules π1–π7 delete
  go with them, so it is amended to the remainder, not cancelled.
- 7838 (pending, low) percent-encodes raw `file:{path}?mode=ro` opens. The runs.db opens in seven scripts move
  onto `shared.runs_read` under σ and τ (`merge_lane_throughput.py::_connect_ro`,
  `suite_census_evidence.py::_connect_ro`, `audit_model_admission.py::connect_ro`,
  `mint_hard_v2_fixtures.py::connect_ro`, `census_tagger_debris.py::_connect_readonly`,
  `audit_wiped_metadata_files.py`, `merge-pytest-n-ab-analysis.py`); 7838 keeps the rest and follows σ and τ
  (lead amend and edges).
- 8661 (pending, low) adds `shared/async_sqlite_base.py::connect_read_only`; it follows β and builds its URI with
  `tasks_db_readonly.read_only_uri` (lead amend and edge).
- 8465 (pending, medium, quality-dashboard ι) adds an instrument marker kind to `quality_markers.py`; ρ3 follows
  it.
- 5717 `get_statuses` docstring and 3396 defensive `get_statuses` pagination: they own the residue of the fixed
  fk-8dfa5e5a75aa.
- 4790 healthz data-plane coverage: deferred; μ keeps it applicable.

Landed since the draft, so nothing is wired onto them: 7612 (`3b1f243a9b`; the draft had α wait on it and the
lead amend set count it) and 8460 (`5a1f289e91`; the draft had a lead edge 8460 → β and an amend, both dropped,
and ρ3 takes its reads). 8461 is in progress with a plan authored, so the draft's amend to it is dropped: its
additions to `test_cache_buster_freshness.py::_REQUIRED_ASSETS` go with ξ2, and every conversion task follows it.

## Out of scope

- Converting the classic scripts to ES modules or adding a build step (ruled out at the sitting, OQ-3).
- The bundle-structure checks the sitting kept (OQ-3): `classic_script_scope.test.mjs` and `test_index_html.py`'s
  structure tests.
- Fused-memory's read model itself (S15 κ).
- The dashboard's MCP egress module (S19 C8).
- Moving runs.db DDL or migrations out of the orchestrator.
- The `plans/` one-shot readers, and the six test-side runs.db DDL copies D3 names (opportunistic).
- Every dashboard reader of escalation statuses (S2 η).
- The task-census walk (fk-8dfa5e5a75aa, fixed; residue owned by 5717 and 3396).
- Retention (landed in `data/retention.py`).

## Rulings (sitting of 2026-11-12)

Leo answered the sitting on 2026-11-12 (`digests/wave3-rulings-applied.md` §S22 in the program's state home);
the lead's rulings on the minor points are in `digests/wave3-lead-rulings.md`. No question of this plan is
pending.

- **OQ-1** (sitting S22-1): move the event and merge-outcome names into the shared package now, with the 22-task
  amend set and the task-authoring pointer, as recommended (D2, α). Ruled by Leo.
- **OQ-2**: yes, the orchestrator's typed event reads run the shared queries (υ, after 9231). Ruled by the lead.
- **OQ-3** (sitting S22-2): revalidation only; the scope test and the tag-order tests stay the enforcement of the
  other two loading rules (D11, ξ1, ξ2). Ruled by Leo.
- **OQ-4** (sitting S22-3): option B, not the recommended frozen exception. The guard plus eight conversion tasks
  filed now: seven by tab and one cleanup, all medium, each tagged with fk-1bdd95e804cb, each writing its own
  `.test.mjs`, each recording the pins it drops; the exemption list is owned debt, shrink-only through the ratchet
  kernel, with no ratification row; the lead cancels nine helper-upkeep tasks, subsumes 7907, 8703 and 7566,
  amends 6963, and edges and amends six feature tasks (D14, C6, ο, π1–π8). Ruled by Leo; the measured "Effect on
  the plan if adopted" of `digests/discussion/S22.md` §S22-3 is part of the ruling.
- **OQ-5**: as S16/OQ-4 option 2, the dashboard imports the shared constants for the escalation port and the
  concurrency cap and restates nothing (D12, λ). Ruled by the lead.
- **OQ-6**: the package-layering plan's `dashboard/data/mcp_client.py`; this plan consumes it (D7). Ruled by the
  lead (as S19/Q-C3).
- **OQ-7**: the five unambiguous table names plus the co-occurrence rule (C5, ω). Ruled by the lead.
- **OQ-8**: `run_query` on the caller's connection (C5, ρ1, ρ3). Ruled by the lead.
- **The guard's population** (after the fold-verifiers): ο's guard covers both clauses, a direct read of a
  frontend source file and a read through the helper, and the conversion tasks take what both clauses find
  (D14 *Scope*: π3, π6, π7, π8 and ξ2); narrowing C6 to the helper clause would let a raw read of a `.jsx`
  through. Ruled by the lead.
- **Test-file edges** (after the fold-verifiers): an edge that rests only on a shared test file is not wired,
  inside the plan or across plans (§Decomposition plan names the three dropped). Ruled by the lead.

No ruling amends a program ruling (R1–R60): R19's "the 73%/63% prose moves to task records" and its remaining
items are carried as before. Two standing rules Leo stated at the sitting are checked against this plan. The
decision-partition invariant, `one-decision-partition-per-area` (INV-23): every decision this plan adds (D5's outage rule, C2a's failure
classification, C6's offender rule) is a code table applied in one place, and no row is handed to an LLM; a
conversion task's choice to drop a pin is authoring judgement recorded in its commit, not a runtime decision.
Hermetic evals: the plan adds no store, sink, filer or queue an eval cell can reach (the read model only reads).

## Tactical choices left to the leaves

- **T2** `OutcomeFamily` member names: take `outcome_colors.py`'s family set at θ's base.
- **T3** η's `_HISTORY_CACHE`: keep it as a max_ts-keyed memo of raw per-DB tallies, or drop it if the
  per-DB read is under 50 ms on the live DB. Measure in η.
- **T4** RECON_STATE's unknown shape: one panel-level Datum or per-reading Datums. Decide in ε by what `tabs.jsx`
  renders most simply.
- **T5** Whether `api/memory_graphs.py` joins the existing `api/memory.py` router. Decide in ν by heuristic 6.
- **T6** `CacheRegistry`'s field names: one per module cache at δ2's base.
