# Capability manifest — S22 dashboard readers and aggregation

PRD: `plans/review-all-s22-dashboard-readers-and-aggregation-prd.md`. YAML twin:
`plans/review-all-s22-dashboard-readers-and-aggregation-prd.capability-manifest.yaml` (every `task_id: null`
until decompose; signal leaves ω and π8). The rulings of the sitting of 2026-11-12 are folded in (PRD §Rulings).

**Code anchors** verified against main `1b106418da` (2026-11-12). Bindings cite symbols, not lines. Every
mechanical producer check below was evaluated at that tree with
`shared/src/shared/delivered_check_polarity.py::evaluate_grep_at_tree` / `evaluate_path_at_tree` and linted with
`lint_delivered_checks` against the leaf's declared files, and re-evaluated at `0f21f5e457` by the fold
verifier. Each of the 107 producer checks fails at that tree, and each passes once its leaf lands. The seven `*-substrate` checks pass today and are never copied into a
task's `metadata.delivered_checks`. One lint warning is expected and not blocking: π8's
`no-source-text-helper-under-tests` matches files outside π8's declared files today, because π1–π7, on which
π8 depends, delete them first. "X → Y" means X depends on Y.

## α — Move the run-event and merge-outcome vocabularies into shared

| Capability | Binding | Verdict |
|---|---|---|
| `EventType` and `OutcomeKind` exist once, in an orchestrator module each | `orchestrator/src/orchestrator/event_store.py::EventType` (130 members), `orchestrator/src/orchestrator/merge_lane/types.py::OutcomeKind` (23), measured at 1b106418da | PASS |
| `shared` may hold a StrEnum vocabulary imported by every package | precedent `shared/src/shared/task_statuses.py::TaskStatus`; shared has no workspace dependency | PASS |
| A second copy to delete | `scripts/merge_lane_throughput.py::NON_TERMINAL_OUTCOMES` (made public by task 7612, landed `3b1f243a9b`), pinned by `orchestrator/tests/test_outcome_kind.py::TestOutcomeKindFrozenContract`, which loads the script by path, measured at 1b106418da | PASS |
| `merge_lane/types.py` is measured by the merge-lane ratchet | `orchestrator/tests/merge_lane_ratchet_baseline.json` lists it; α re-measures the baseline | PASS |
| Produced: `shared/run_events.py`, `shared/merge_outcomes.py`, the task-authoring pointer | α; checks `run-events-module`, `event-type-defined-once`, `outcome-kind-defined-once`, `script-non-terminal-copy-gone`, `authoring-doc-pointer` | PASS (α delivers) |
| Position on the wave-2 chains (α → 9140; 9152 → α; 9144 → α) and the 22 amends | ruled at the sitting (OQ-1, Leo); wired by the lead at decompose; manual `wave2-chain-position` | PASS |

## β — Shared runs.db read model

| Capability | Binding | Verdict |
|---|---|---|
| runs.db locator | `shared/src/shared/project_layout.py::runs_db` (task 6710) — check `runs-db-locator-substrate` | PASS (substrate) |
| Counted-outcome source | `shared/src/shared/task_transitions.py::WORKFLOW_OUTCOMES` — check `workflow-outcomes-substrate` | PASS (substrate) |
| The read-only opener five runs.db readers already use | `shared/src/shared/tasks_db_readonly.py::connect_ro`, `TaskDbProblem`, `TaskDbUnreadable` — check `read-only-opener-substrate` | PASS (substrate) |
| `aiosqlite` and `sqlite3` available to `shared` | `shared/pyproject.toml` runtime deps include `aiosqlite` | PASS |
| Real writers to drive the contract test | `orchestrator/.../event_store.py::EventStore`, `run_store.py::RunStore`, `shared/cost_store.py::CostStore`, `flake_ledger.py` (DDL owners of runs.db's 12 tables, measured) | PASS |
| Produced: `RunsQuery` (with `requires_columns`/`fallback`), `Read`/`Absent`/`Unreadable`, `read_runs_db`, `run_query`, `read_runs_db_async`, `read_only_uri`, `COUNTED_OUTCOMES`, fixture writer, schema floor, orchestrator contract test | β; checks `runs-read-query-module`, `unreadable-is-a-value`, `executor-on-a-connection`, `read-only-uri-builder`, `counted-outcomes-home`, `producer-contract-test`, `schema-floor-fixture` | PASS (β delivers) |

## γ1 — One fan-out failure classification; memory's queue and WAL reads fail over

| Capability | Binding | Verdict |
|---|---|---|
| httpx exception hierarchy | httpx 0.28.1 in the root venv: `TimeoutException` ⊂ `TransportError`; `ReadError`/`RemoteProtocolError` ⊄ the current tuple (measured) | PASS |
| Failover topology for memory's queue/WAL reads | `DESIGN.md` "DASHBOARD_FUSED_MEMORY_URLS is failover-only, not sharding" | PASS |
| The dashboard's MCP egress below `memory` and `mcp_fanout` | producer: S19 κ1 (S19 C8, `dashboard/data/mcp_client.py`; lead ruling S19/Q-C3 = OQ-6), unfiled at authoring — manual `s19-egress-delivered` | OPEN (cross-plan edge at decompose) |
| Produced: `classify_fanout_failure`; URL loops gone | γ1; checks `one-classifier`, `hand-rolled-url-loops-gone` | PASS (γ1 delivers) |

## γ2 — One concurrent escalation-URL probe

| Capability | Binding | Verdict |
|---|---|---|
| Base-URL rule's one home | `shared/src/shared/mcp_post.py::mcp_server_base` (task 6747) — check `mcp-server-base-substrate` | PASS (substrate) |
| Per-key transition logging primitive | `shared` `KeyedStreaks` (used by mcp_fanout since 2be50e76bf) | PASS |
| Produced: `probe_escalation_urls`; four `removesuffix('/mcp')` copies gone | γ2; checks `probe-helper`, `base-url-copies-gone` | PASS (γ2 delivers) |
| Rejection: one orchestrator's `ReadError` leaves the merge-queue endpoint 200 | boundary row 2, manual | PASS (γ2 delivers) |

## δ1 — TTLCache in its own module, one slot per key

| Capability | Binding | Verdict |
|---|---|---|
| Reaper primitive below the cache | `dashboard/data/mcp_fanout.py::cancel_and_await` (downward import, no reach-back; hotspot h14 measurement) | PASS |
| Produced: `ttl_cache.py`; mcp_fanout holds no cache; no `_waiters`; lineage on C3; burndown off mcp_fanout's constant | δ1; checks `ttl-cache-module`, `fanout-holds-no-cache`, `no-asyncio-private-read`, `lineage-single-flight-gone`, `burndown-off-fanout-constant` | PASS (δ1 delivers) |

## δ2 — App-scoped caches; tests construct policies

| Capability | Binding | Verdict |
|---|---|---|
| A per-test fixture home | `dashboard/tests/conftest.py` (exists) | PASS |
| Produced: `CacheRegistry`; callable TTL gone; no test patches a TTL constant | δ2; checks `cache-registry`, `callable-ttl-gone`, `tests-construct-policy` | PASS (δ2 delivers) |

## ε — Recon and memory-ops panels via fused-memory's read model

| Capability | Binding | Verdict |
|---|---|---|
| fused-memory serves the panels | producer: S15 κ (S15 C4, `shared/dashboard_panels.py`, `get_status(panels=…)`), unfiled at authoring | OPEN (cross-plan edge at decompose; manual check `s15-read-model-delivered`) |
| Produced: recon SQL (incl. `FROM runs`), index hint, DDL fixtures and the write-journal path gone | ε; checks `recon-sql-gone`, `index-hint-gone`, `recon-ddl-fixtures-gone`, `journal-db-path-gone` | PASS (ε delivers) |
| The burst state's derived reading served by the owner | S15 C4 `BurstStateRow.effective_state` (computed with the instance's `burst_cooldown_seconds`); ε deletes the reader's copy, check `reader-cooldown-copy-gone` | PASS (ε delivers) |

## ζ — Curator panel honest about outages; samplers write NULL

| Capability | Binding | Verdict |
|---|---|---|
| `Datum`/`unknown_datum` to carry unknown | `dashboard/src/dashboard/data/datum.py` (measured; 30 modules use it) | PASS |
| `CuratorPanel` (resolutions, refused_count) | S15 C4 (same producer row as ε) | OPEN (via ε's row) |
| Produced: zero substitution, tickets SQL and the tickets.db plumbing gone | ζ; checks `zero-substitution-gone`, `tickets-sql-gone`, `tickets-db-plumbing-gone` | PASS (ζ delivers) |
| A tick with fused-memory stopped writes NULL | live metrics.db read, manual `outage-writes-null` | PASS (ζ delivers) |

## η — Performance: additive tallies, one merge

| Capability | Binding | Verdict |
|---|---|---|
| Two-DB fixtures for the same project | `dashboard/tests/test_performance.py` fixtures `two_project_conns`, `two_hist_conns` (measured) | PASS |
| Produced: `merge_tallies`; `_merge_history`, `get_time_centiles` and the patch sites gone | η; checks `one-merge`, `percentile-mean-gone`, `dead-centiles-gone`, `unreachable-branch-patches-gone` | PASS (η delivers) |

## θ — Dashboard event readers on the read model

| Capability | Binding | Verdict |
|---|---|---|
| The quality access-path guard to keep honest | `dashboard/tests/test_datum_access_paths.py::_QUALITY_GRANTS` (three `data/merge_queue.py` grants), `::GUARDED_QUALITY_READS` (measured at 1b106418da) | PASS |
| Produced: `_CODE_FAMILY` gone and `OUTCOME_FAMILY` in its place; event literals (incl. `tabs.jsx`), `MERGE_EVENTS_SCHEMA`, `_runs_db_fixture.py` and the merge_queue grants gone | θ; checks `code-family-copy-gone`, `outcome-family-in-dashboard`, `merge-event-literals-gone`, `test-events-ddl-copy-gone`, `dashboard-runs-fixture-copy-gone`, `merge-queue-quality-grants-retired` | PASS (θ delivers) |

## ι — Cost, cap, model-role and load readers typed; with_db deleted

| Capability | Binding | Verdict |
|---|---|---|
| Frontend renders null as "—" | `dashboard/static/redux/spend_readings.js::todaySpend` → `plainDatum` returns UNKNOWN for null (measured) | PASS |
| Per-attempt `invocations` rows | producer: S20 ζ (S20 C7), unfiled at authoring; ι → S20 ζ wired at decompose | PASS (dependency, not a check) |
| Produced: `with_db` gone; `shape_costs` null; `AccountEventType`; the model-role query module | ι; checks `with-db-gone`, `spend-today-not-synthetic`, `account-event-vocabulary`, `model-role-query-module` | PASS (ι delivers) |

## κ — Scheduler collector record and raw task ids

| Capability | Binding | Verdict |
|---|---|---|
| Produced: `SchedulerState`; uid parses and stale rationale gone | κ; checks `named-scheduler-record`, `uid-parse-gone`, `stale-rationale-gone` | PASS (κ delivers) |

## λ — Orchestrator discovery and orchestrator-yaml read

| Capability | Binding | Verdict |
|---|---|---|
| Orchestrator defaults to agree with | `orchestrator/src/orchestrator/config.py::EscalationConfig.port` default 8100 (measured at 1b106418da); `max_concurrent_tasks` (S16 α moves it to `shared/src/shared/config_defaults.py::DEFAULT_MAX_CONCURRENT_TASKS`, S16 C8) | PASS |
| The cross-package defaults module | producer: S16 α (S16 C8, `shared/src/shared/config_defaults.py`), unfiled at authoring; λ → S16 α wired at decompose | PASS (dependency, not a check) |
| Produced: `RunningOrchestrator`; the mirrored discovery gone | λ; checks `typed-discovery-record`, `mirrored-discovery-gone` | PASS (λ delivers) |
| Default sourcing (lead rulings S16/OQ-4 option 2 and OQ-5): the port's default in C8, named by `EscalationConfig.port` | λ; checks `escalation-port-shared-default`, `escalation-port-field-names-constant` | PASS (λ delivers) |

## μ — /healthz HealthProbe on app.state

| Capability | Binding | Verdict |
|---|---|---|
| Produced: `HealthBudget`; module globals and reset hook gone | μ; checks `health-module`, `module-globals-gone`, `test-reset-hook-gone` | PASS (μ delivers) |

## ν — Routes into api/ routers with typed bodies

| Capability | Binding | Verdict |
|---|---|---|
| Partition with no reach-back | heuristic-14 measurement in fk-ce66d4da9b0f, re-measured at 1b106418da (healthz 468, GET 423, proxies 338, lifespan 314, wiring 35 lines) | PASS |
| Tier vocabulary's home | `shared/task_priority.py::PRIORITY_TIERS` from S14 θ 6718 (pending); ν depends on 6722 through μ, which depends on 6718 | PASS (upstream S14) |
| The underscore names the routers import, made public | producer: S19 κ2, unfiled at authoring; ν → S19 κ2 wired at decompose | PASS (dependency, not a check) |
| Produced: `api/scheduler.py`; no `@app.post`; no `_VALID_BOOST_TIERS` | ν; checks `scheduler-router`, `no-post-routes-in-app`, `boost-tier-copy-gone` | PASS (ν delivers) |

## ξ1 — Static assets revalidate; the tests stop pinning a `?v=` floor

| Capability | Binding | Verdict |
|---|---|---|
| `StaticFiles` sends ETag/Last-Modified and answers a conditional GET with 304 | starlette `StaticFiles` (installed with FastAPI in the dashboard env) | PASS |
| The JS harness loads unstamped tags | `dashboard/tests/js/_served_bundle.mjs` `CLASSIC_SCRIPT_RE` and `BABEL_SCRIPT_RE` make the query optional (`(?:\?[^"]*)?`, measured) | PASS |
| Produced: the revalidating static mount; the floor pins gone | ξ1; checks `static-revalidation-module`, `cache-buster-floor-pins-gone` | PASS (ξ1 delivers) |

## ξ2 — Strip the stamps; the freshness test goes

| Capability | Binding | Verdict |
|---|---|---|
| Produced: `?v=` literals, the freshness test and its helpers gone | ξ2; checks `version-literals-gone`, `freshness-test-gone`, `buster-helpers-gone` | PASS (ξ2 delivers) |
| Produced: the four served-module body checks move to node before ο lands, so they are never listed (D14 Scope) | ξ2; checks `served-body-substring-checks-gone`, `served-module-globals-node-test` | PASS (ξ2 delivers) |
| Classic-script scope and load-order invariants | stay enforced by `dashboard/tests/js/classic_script_scope.test.mjs` and `dashboard/tests/test_index_html.py` (OQ-3, ruled at the sitting) — check `scope-and-order-tests-substrate` | PASS (substrate) |

## ο — Guard: no new dashboard test reads frontend source text; each exempted module is owned by its conversion task

| Capability | Binding | Verdict |
|---|---|---|
| A DOM harness to point new tests at | `dashboard/tests/js/dom_harness.mjs::buildDashboardWindow`, `jsx_render.test.mjs` (measured) | PASS |
| A comparison under which the list only falls | `shared/src/shared/ratchet.py::excess`, `::slack` — check `ratchet-kernel-substrate` | PASS (substrate) |
| An owner on every exempted module (INV-12 owned debt) | `shared/src/shared/governed_exceptions.py::governed_exceptions`, `Debt`, `TaskRef` (precedent `dashboard/tests/test_dashboard_governed_lists.py`) | PASS |
| Produced: the guard, its seeded fixture and the owned exemption list (C6) | ο; checks `source-text-guard`, `exemptions-owned-debt` | PASS (ο delivers) |
| No ratification row | option B ruled at the sitting (OQ-4): every entry is `Debt(owner)` of a live conversion task | PASS |

## π1–π7 — The tab groups' source-text tests become render tests

| Capability | Binding | Verdict |
|---|---|---|
| A harness that renders any tab | `dashboard/tests/js/dom_harness.mjs::buildDashboardWindow` (`mount`, `mountPage`); `jsx_render.test.mjs` mounts the Tasks, Escalations, Memory-evals, Merge and Performance tabs, the app shell and the error boundary (79 tests in 9.7 s, measured 2026-11-10) | PASS |
| A new `.test.mjs` runs with no registration | `dashboard/tests/test_graph_layout_js.py::_JS_TESTS_GLOB` (`**/*.test.mjs`, measured at 1b106418da) | PASS |
| Fixture builders for the wire shapes | `dashboard/tests/js/wire_fixtures.mjs` (measured) | PASS |
| A node loader for `data.js` (π7's abort-delay probe) | `dashboard/tests/js/_data_loader.mjs` (measured) | PASS |
| The ruled population, by tab group | 30 modules, 397 tests, measured at 1b106418da with the sitting's grep rule; plus what C6's guard also finds (re-measured at 0f21f5e457, D14 Scope): `test_charts_consumer_bindings.py` (π3), the two `_browser_abort_ms` reads (π7), and the two importers of `_lock_chip_matrix.py`, `test_lock_chip_state.py` and `test_boundary_js.py` (π6) | PASS |
| Produced, per task: its render-test file; its modules gone | π1 `render-tasks-tab-file`, `π1-source-text-modules-gone`; π2 `render-overview-orchestrators-file`, `π2-source-text-modules-gone`; π3 `render-burndown-charts-file`, `π3-source-text-modules-gone`; π4 `render-memory-tabs-file`, `π4-source-text-modules-gone`; π5 `render-escalation-tabs-file`, `π5-source-text-modules-gone`; π6 `render-ops-tabs-file`, `π6-source-text-modules-gone`, `lock-chip-matrix-importers-rewired`; π7 `render-app-shell-file`, `π7-source-text-modules-gone`, `abort-timeout-not-read-from-source` | PASS (each task delivers its own) |
| Every dropped fidelity pin recorded with its reason | the task's commit message (D14); a reviewer reads it at review | PASS (each task delivers) |

## π8 — Delete the source-text helper layer; the exemption list ends empty

| Capability | Binding | Verdict |
|---|---|---|
| The bundle reader the last served-script checks move onto | `dashboard/tests/js/_served_bundle.mjs`, as `classic_script_scope.test.mjs` uses it (measured) | PASS |
| Produced: no source-text helper named under `dashboard/tests` (the sitting's delivered check); the helper modules gone; conftest's nine served-script fixtures gone; the retired-globals node test | π8; checks `no-source-text-helper-under-tests`, `helper-modules-gone`, `served-script-fixtures-gone`, `retired-globals-node-test` | PASS (π8 delivers) |

## ρ1 — Quality series on the read model

| Capability | Binding | Verdict |
|---|---|---|
| The connection-taking readers to keep | `shared/src/shared/quality_series/landings.py::landings`, `::recorded_landing_keys`, `shared/src/shared/quality_series/rows.py::rows_for` (each takes a `sqlite3.Connection`) and `::_has_charter_column` (measured) | PASS |
| An executor on a caller's connection | β's `run_query` (lead ruling OQ-8) | PASS (β delivers) |
| Produced: raw SQL gone from `landings.py` and `rows.py`; the quality query module | ρ1; checks `quality-landings-sql-gone`, `quality-rows-sql-gone`, `quality-queries-module` | PASS (ρ1 delivers) |

## ρ2 — Client feed and fused-memory's scheduler reads on the read model

| Capability | Binding | Verdict |
|---|---|---|
| The shared scheduler-state reader to migrate | producer: task 8559 (`shared/scheduler_state.py::read_scheduler_state_checked`), pending; ρ2 → 8559 | PASS (dependency, not a check) |
| Produced: `StoreEventType` gone; raw `FROM events` gone from `scheduler_state.py`; the scheduler-row query module | ρ2; checks `client-feed-mirror-gone`, `scheduler-events-sql-gone`, `scheduler-row-query-module` | PASS (ρ2 delivers) |

## ρ3 — Quality markers read runs.db through the read model

| Capability | Binding | Verdict |
|---|---|---|
| The raw reads to move | `dashboard/src/dashboard/data/quality_markers.py::fold_routing_decisions` (`_ROUTING_SQL`), `::read_config_changes` (`_CONFIG_RELOAD_SQL`), `::deploy_markers`, `::model_version_markers`; opened by `quality_marker_sampler.py::_refresh_root` (task 8460, `5a1f289e91`; measured at 1b106418da) | PASS |
| An executor on a caller's connection and column requirements with a fallback | β's `run_query`, `requires_columns`, `fallback` (C5, D4b) | PASS (β delivers) |
| Per-attempt `invocations` rows to classify | producer: S20 ζ (S20 C7), unfiled at authoring; ρ3 → S20 ζ wired at decompose | PASS (dependency, not a check) |
| Produced: the markers query module; raw SQL gone from `quality_markers.py`; the sampler opens through `open_runs_db` | ρ3; checks `markers-queries-module`, `markers-sql-gone`, `markers-sampler-opens-runs-db` | PASS (ρ3 delivers) |

## σ — Merge-lane scripts on the read model

| Capability | Binding | Verdict |
|---|---|---|
| Scripts import `shared` under the system python | `scripts/_checkout.py` bootstrap (S25 γ 6758, done) | PASS |
| Produced: private openers and the `PLAN_EVENT_TYPE` copy gone | σ; checks `private-openers-gone`, `plan-event-copy-gone` | PASS (σ delivers) |

## τ — Audit and operator scripts on the read model

| Capability | Binding | Verdict |
|---|---|---|
| `query-events.sh` is unreferenced | `git grep -l 'query-events.sh'` finds only itself (measured) | PASS |
| Produced: shell reader, digest import, audit opener gone; the audit query module | τ; checks `unreferenced-shell-reader-gone`, `fleet-state-off-digest`, `audit-opener-gone`, `audit-queries-module` | PASS (τ delivers) |

## υ — EventStore reads delegate to the read model

| Capability | Binding | Verdict |
|---|---|---|
| EventStore typed reads exist to delegate | producer: S8 ι2 9231 (pending), upstream of υ | OPEN (dependency wired at decompose) |
| Produced: `event_store.py` imports `shared.runs_read`; no raw events SQL in `event_store.py` or `digest.py` | υ; checks `event-store-delegates`, `in-process-events-sql-delegated` | PASS (υ delivers) |
| The digest and the dashboard run one model-role query (OQ-2, ruled by the lead) | boundary row 19, manual `one-model-role-query` | PASS (υ delivers) |

## ω — Integration gate and the encapsulation guard

| Capability | Binding | Verdict |
|---|---|---|
| C1 `ProjectLabels.root_for` | `dashboard/src/dashboard/project_labels.py` (task 5760), measured at 1b106418da — check `project-labels-substrate` | PASS (substrate) |
| The guard's roots | `pyproject.toml` `[tool.uv.workspace].members` (8 members, measured) plus `scripts/` | PASS |
| Produced: encapsulation guard (no list; co-occurrence rule, lead ruling OQ-7) and boundary suite (rows 1–18, 20–22, 24) | ω; checks `encapsulation-guard`, `boundary-suite` | PASS (ω delivers) |
| §Goal holds on the live dashboard | manual `goal-holds-live` | PASS (ω delivers) |
