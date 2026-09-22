# Capability Manifest — dashboard-one-datum-one-path-prd

Mechanizes G3 (substrate exists/**wired**) + G6 (premise valid) per task. One block
per task; each capability bound to on-main evidence. Any **FAIL** binding blocks
queueing.

**Domain flags:** tooling/infra domain — no grammar/DSL → grammar-fixture checks
**N/A**; the only numeric premises are measured costs (status map 0.07–0.15 s, active
rows ~1 s, whole tree 3.4 s / 29 MB) and they bound nothing a test asserts →
numeric-floor checks **N/A**. Live checks: **capability→producer (wired)**,
**DAG-direction**, **field-population** (the `Datum` invariants), **rejection-mechanism**
(the deletions and the non-`Datum` throw).

Evidence re-confirmed against main `1d49edcd9b` (2026-09-18) during the decompose
re-walk; cited by symbol. This PRD introduces no novel substrate: the `Datum` type,
`census.py`, `task_snapshot.py`, `task_lookup.py`, the generator and the AST check are
its own deliverables, each bound below to the leaf that builds it.

**Delivered-check gate is LIVE**: `commit_planning` stamps the YAML sidecar and copies
each producer's **mechanical** checks into `metadata.delivered_checks`. Mechanical
checks are bound conservatively — `def <name>` / symbol presence greps on the
producer's own new module — so a consumer can never be wedged by a check that a
comment could satisfy or a refactor could false-negative. Deletions, JSX rendering
and the boundary suite are `kind: manual`.

Machine-readable twin: `plans/dashboard-one-datum-one-path-prd.capability-manifest.yaml`.

---

## α — `Datum` envelope, `TaskCensus` + views, generated vocabulary *(intermediate → β, γ1)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Closed nine-member vocabulary with derived `ACTIVE`/`TERMINAL` | capability→producer (wired) | `shared/src/shared/task_statuses.py::TaskStatus`, `::ACTIVE = frozenset(TaskStatus) - TERMINAL`; dashboard depends on `dark-factory-shared` (`dashboard/pyproject.toml`) and four dashboard tests already import `shared.*` | PASS |
| Three-view partition + `running` sub-view is arithmetically consistent | field-population | `in_flight` (5) + `backlog` (2) + `terminal` (2) = 9 members, disjoint; `running = {IN_PROGRESS} ⊆ in_flight`; wire splits `views`/`sub_views` so `sum(views) == total` | PASS (built+bound by α) |
| Generator + parity-test precedent in root `tests/scripts/` | capability→producer (wired) | `scripts/render_dashboard_unit.py` + `tests/scripts/test_dashboard_service_template.py::test_template_renders_to_hardcoded_file`; `scripts/` on `sys.path` there | PASS |
| Classic-script load order + window-global uniqueness | capability→producer (wired) | `static/redux/index.html` classic `<script>` tags before Babel; `tests/js/classic_script_scope.test.mjs::EXPECTED_WINDOW_GLOBALS` enumerates globals — α adds `DF_TASK_VOCAB` there | PASS |

## α2 — Extract handlers + the two loops from `app.py` *(intermediate → β)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Handlers and loops exist to move | capability→producer (wired) | `app.py::api_tasks`, `::api_orchestrators`, `::api_burndown`, `::api_merge_queue`, `::api_escalations`, `::api_memory`, `::_parse_window`, `::_burndown_loop`, `::_metrics_loop` | PASS |
| Healthz probe keeps its raw fetch | rejection-mechanism | `app.py::_fanout_probe_completion` calls `fetch_tasks` uncached by design (docstring); α2 moves it unchanged and decision 17 exempts it | PASS (bound by α2, verified by ι) |
| Behaviour unchanged | field-population | every `test_app.py` assertion retained with retargeted patch paths | PASS (built+bound by α2) |

## β — One task snapshot unit on `/tasks`; `/orchestrators` loses `summary` *(intermediate → γ2, γ3, δ1)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Paged compact status map | capability→producer (wired) | fused-memory `get_statuses` `page_size`/`offset`/`has_more` (called live: 5539 statuses in three pages); today's `tasks.py::fetch_statuses` is unpaginated — β wires paging | PASS |
| Server-side `statuses` filter accepts `review`/`infra-hold` | capability→producer (wired) | `fused_memory/server/tools.py::_VALID_TASK_STATUSES = ACTIVE_TASK_STATUSES \| TERMINAL_STATUSES` from `shared`; memory `fa730a43` (real `WHERE status IN`) | PASS |
| Live/stranded split from rows | capability→producer (wired) | `shared/src/shared/task_claimant.py::is_stranded`; `tasks.py::task_is_stranded` | PASS |
| α's census upstream | DAG-direction | β depends on α | PASS |
| Last-good served as `stale` on a transient (sketch #13) | field-population | producer = β; `Datum` validator asserts `state=="stale" ⇒ reason` — a transient can never yield zeros | PASS (built+bound by β) |
| Terminal rows on demand | capability→producer (wired) | `active_tasks.py::_TERMINAL_FETCH_WINDOW` + `tasks.py::fetch_task_page(statuses=…)` exist; β moves the call behind `?terminal=` | PASS |

## γ1 — Shared components render a `Datum`; registry *(intermediate → γ2, γ3, δ2, ζ, η, θ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Components to extend | capability→producer (wired) | `charts.jsx::StatTile`, `tabs.jsx::ST`, `shell.jsx::ProjectGroup`, `tabs.jsx::LocksCell`, `task_row_cells.js` | PASS |
| Registry extension point + receipt metadata | capability→producer (wired) | `data.js::applyKey`, `::__loaded`, `endpointsFor`; endpoint freshness stays with `endpoint_staleness.js::STALE_FAILURE_THRESHOLD` / `data.js::__stale` / `app.jsx::staleNoticesForTab` (task 4884) — γ1 adds `_served_at`/`_received_at`, never a state | PASS |
| Non-`Datum` throws under the node harness | rejection-mechanism | `tests/js/*.test.mjs` run via `dashboard/tests/test_graph_layout_js.py::test_graph_layout_js_suite_passes` (in the merge gate) — γ1 adds the throwing assertion and its test | PASS (built+bound by γ1) |

## γ2 — Orchestrators tab, Overview, app chrome consume the census *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Census on the wire (upstream) | DAG-direction | β upstream; `TASKS_SNAPSHOT[p].census.views` | PASS |
| Components accept a `Datum` (upstream) | DAG-direction | γ1 upstream | PASS |
| "0/1" cannot render | rejection-mechanism | `tabs.jsx::OrchTab` `total \|\| 1` and `o.summary` deleted; sketch #1–#3 fixtures under the node harness | PASS (built+bound by γ2, verified by ι) |

## γ3 — Tasks tab, PRD boxes, on-demand terminal fetch, bucketers deleted, 4416 landed *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `task_status_counts.js` and its four pins | capability→producer (wired) | `static/redux/index.html` script tag; `test_index_html.py::test_task_status_counts_js_is_served` / `::..._loads_before_tab_tasks`; `test_cache_buster_freshness.py::_REQUIRED_ASSETS`; `tests/js/classic_script_scope.test.mjs::EXPECTED_WINDOW_GLOBALS` — all named in γ3 | PASS |
| 4416's contract is decided here | capability→producer | task 4416 (pending) option (a) + `lower_bound`; γ3 amends `plans/dashboard-taskgraph-legibility-prd.md`; 4416 wired to depend on γ3 | PASS |
| `?terminal=` server half upstream | DAG-direction | β upstream (γ3 depends on γ2 → β) | PASS |

## δ1 — Burndown sampler consumes the snapshot unit; gap rows; nullable columns *(intermediate → δ2)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Nullable column migration idiom | capability→producer (wired) | `burndown.py::ensure_snapshot_columns` + `_ADDED_SNAPSHOT_COLUMNS` (`concurrency_cap INTEGER` nullable precedent); the `NOT NULL DEFAULT 0` form is explicitly rejected | PASS |
| Snapshot unit upstream | DAG-direction | β upstream | PASS |
| Loop lives in `loops.py` | DAG-direction | α2 upstream (via β) | PASS |
| Gap row on failure, value row on success (deterministic fixture) | field-population | producer = δ1; asserted against a fixture store, not live rate | PASS (built+bound by δ1) |

## δ2 — Carry-last aggregate, staleness on the wire, parity/forecast over measured rows *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Aggregate builder + parity/forecast to rewrite | capability→producer (wired) | `redux_api.py::shape_burndown`, `::_aggregate_parity`, `burndown.py::compute_parity_alarm`, `compute_forecast_confidence` | PASS |
| Ragged test is updated, not deleted | rejection-mechanism | `test_redux_api.py::test_shape_burndown_ragged_input_is_passed_through_unnormalized` docstring pre-authorises "updated to the new behaviour, not deleted" | PASS |
| `dailyDeltas` deletion | rejection-mechanism | `shell.jsx::dailyDeltas` sole consumer is OrchTab's spark; server `completed` series exists (`burndown.py::compute_window_completion`) | PASS (verified by ι) |

## ε1 — Window echo, per-tab validation, labels from payload, recent merges follow the chip *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Window parsing and chip state | capability→producer (wired) | `app.py::_parse_window`, `_WINDOW_DAYS`; `app.jsx` `win` state + `Toolbar` `windows=` | PASS |
| Recent merges trim | capability→producer (wired) | `merge_queue.py::build_per_project_merge_queue(recent_window_minutes=…)`; `test_app.py` 1440 pin retired (smallest chip = 24h, behaviour can only widen) | PASS |

## ε2 — Performance wall-clock + idle state; model-role bounds *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Two cutoff conventions to unify | capability→producer (wired) | `performance.py::_project_cutoffs` / `_WINDOW_SQL` vs the hour-bucketed history's wall-clock cutoff | PASS |
| Digest bound semantics to mirror | capability→producer (wired) | `orchestrator/src/orchestrator/digest.py::model_role_rollup`; `model_role.py::_cutoff` (mirror, no import) | PASS |

## ζ — Merge attempts one datum; live "In queue now"; `task_lookup` *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Two merge queries to unify | capability→producer (wired) | `merge_queue.py::_get_durations` (`duration_ms > 0`) vs `::outcome_distribution` | PASS |
| Live value + sampled history | capability→producer (wired) | `merge_queue.py::resolve_active`; `metrics.py::get_merge_active_series` over `merge_snapshots` | PASS |
| Per-id read for misses | capability→producer (wired) | fused-memory `tools.py::get_task`; `get_tasks` has **no** id filter (grep 2026-09-18) — hence rows-first + per-id miss path | PASS |
| Snapshot rows upstream | DAG-direction | β upstream | PASS |

## η — Escalations one corpus walk; complete classes; cards read `task_lookup` *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Corpus iterator + classifier + full class list | capability→producer (wired) | `escalation.queue.iter_all_escalation_paths`, `escalation.classify.effective_benign`, `escalation.models.RESOLUTION_CLASSES` — imported by `escalation_analytics.py` today | PASS |
| Cards carry whole rows | field-population | `redux_api.py::shape_escalations` emits `task: dict`; `task_lookup` (ζ, upstream) returns rows | PASS |
| Two loaders to collapse | capability→producer (wired) | `escalations.py::load_queue_escalations` (root) vs `escalation_analytics.py::_load_escalation_records` (root+archive) | PASS |

## θ — Memory ops one query; queue normaliser; recon rule; single lock classifier *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Two ops queries | capability→producer (wired) | `write_journal.py::get_memory_timeseries` (`kind IN`) vs `::get_operations_breakdown` (`GROUP BY operation`) | PASS |
| Queue offline branch | capability→producer (wired) | `redux_api.py::shape_memory` offline early-return; `queue.offline` emitted, unread | PASS |
| Lock classifiers | capability→producer (wired) | `scheduler_utils.jsx::lockChipState`, `scheduler_heatmap.jsx::cellStateFor`, `tabs.jsx::LocksCell`; `scheduler.py` `offline_projects` | PASS |

## ι — Integration gate *(leaf; C-as-integration-gate)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| All producers upstream | DAG-direction | ι depends on θ → … → α, α2 | PASS |
| AST guard apparatus with exemptions | capability→producer (wired) | `dashboard/tests/test_clock_discipline.py::find_clock_violations` (typed `# clock-exempt:` allowlist, matcher tests) — the pattern ι reuses; exemption: `_fanout_probe_completion` | PASS |
| Cold-render measurement is recorded, not gated | field-population | sketch #15; baseline 6.8 s / 16.4 MB (2026-09-18) | PASS |
