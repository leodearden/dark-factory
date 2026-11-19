# Capability Manifest — quality-dashboard-prd

Mechanizes G3 (substrate exists and is **wired**) + G6 (premise valid) per leaf of
`plans/quality-dashboard-prd.md`. One block per task; each capability bound to on-main evidence. Any
**FAIL** binding blocks queueing. Machine-readable twin: `plans/quality-dashboard-prd.capability-manifest.yaml`
(stamped by `commit_planning`).

**Domain flags:** tooling/infra — no grammar/DSL, so grammar-fixture checks **N/A**; the only numeric
bounds are statistical design constants (h = 5, λ = 0.05, n_base = 200) whose basis is the design note's
simulated ARL table (`followups/quality-dashboard-seats/S6.md` §4), not an accuracy claim, so numeric-floor
checks **N/A**. Live checks: **capability→producer (wired)**, **DAG-direction**, **field-population**,
**rejection-mechanism**, **readers-prove-their-producer** (INV-13).

**Code anchors** verified against main `a26991efc9` (2026-11-04) by symbol; re-locate at implementation
time. Evidence greps were run in the decompose session; `0 on main` means the delivered-check pattern has no
match today (polarity: fails now, passes when the leaf lands).

---

## α — Shared producer: landings, verify units, per-landing rows, statistics  *(intermediate → β; also a CLI leaf signal)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `merge_finalized done` rows carry `merge_sha` and `train_merged` rows carry `member_task_ids` + `merge_commit_sha` | field-population (wired) | `orchestrator/src/orchestrator/event_store.py::EventType.merge_finalized` / `::train_merged`; emit sites `merge_lane/intake.py::enqueue_merge_request` done-callback and `merge_lane/worker.py::_COALESCE_TRAIN_ID_PREFIX`; measured df W40–W44: 898 done rows all with sha, 74 train_merged rows of 2–3 members | PASS |
| `merge_blocked.$.category`, `review_completed.$.has_blocking_issues/cycle`, `merge_verify` rows, `flake_occurrence.verdict/detail` exist in production | capability→producer (wired) | `event_store.py::EventType.{merge_blocked, review_completed, merge_verify}`, `review_outcome.py::review_completed_payload`, `flake_ledger.py::FlakeVerdict`; rows measured in S1 §1 | PASS |
| `invocations` rows join by numeric `task_id`; `invocation_end.$.turns` present since 2026-04-09 | field-population | S2 §1.1–1.2 (938 null / 11 non-numeric task ids excluded by rule) | PASS |
| Disk review counter readable for backfill | capability→producer | `orchestrator/src/orchestrator/artifacts.py::get_review_cycles_total`; file at `.worktrees/.task-meta/<id>/review_state.json` (S2 §1.4, 569/630 present) | PASS |
| git first-parent lines/paths per landing within a budget | capability→producer | `git diff-tree --numstat <sha>^ <sha>` (S2 §1.1; S4 measured 23.8 s for 90 days, per-merge cost unmeasured → open question 3) | PASS |
| Pure statistics implementable in stdlib | capability→producer | S6 §Implementation sketch (21 functions, no numpy); numpy is not a `dashboard/pyproject.toml` dependency | PASS |

## β — quality.db store and the 600 s sampler loop  *(intermediate → γ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| A value-or-gap sampler store pattern to copy | capability→producer (wired) | `dashboard/src/dashboard/loops.py::_BurndownStore`, `::_burndown_loop`; `data/burndown.py::_gap_row`, `SnapshotState.GAP` | PASS |
| Loop registry the lifespan test pins | capability→producer (wired) | `dashboard/tests/test_lifespan_samplers.py::_LOOPS` (`'_burndown_loop', '_metrics_loop', 'reaper_loop'`) | PASS |
| Retention excludes the new store | rejection-mechanism | `dashboard/src/dashboard/data/retention.py::apply_retention` is driven by `metrics.py::_DOWNSAMPLE_TABLES`; β adds a test pinning `quality.db` absent from it | PASS (built + bound by β) |
| Per-project runs.db fan-out | capability→producer (wired) | `dashboard/src/dashboard/project_dbs.py` (`_cost_dbs`, `_cost_sources`) over `config.all_project_roots`; `data/db.py::DbPool` | PASS |
| α's producer importable from the dashboard | DAG-direction | α upstream of β; `shared` is a permitted dashboard import (one-datum PRD decision 19) | PASS |

## γ — Reader, endpoint, window vocabulary, access-path guard  *(intermediate → δ, ι; leaf signal on the endpoint)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `Datum` envelope, `DatumState`, `validate_datum`, `WireShaped` | capability→producer (wired) | `dashboard/src/dashboard/data/datum.py::Datum`, `::DatumState`, `::WireShaped` | PASS |
| Window datum echo | capability→producer (wired) | `dashboard/src/dashboard/api/window.py::with_window`, `_parse_window`; `api/burndown.py::_BURNDOWN_WINDOWS` precedent for a tab vocabulary | PASS |
| Router registration pattern | capability→producer (wired) | `dashboard/src/dashboard/api/burndown.py` + `app.include_router`; `dashboard/tests/test_app_module_boundaries.py` | PASS |
| Access-path guard extendable | capability→producer (wired) | `dashboard/tests/test_datum_access_paths.py::GUARDED` | PASS |
| "No producer has written here" state for `charter_class` / `prompt_digest` series | readers-prove-their-producer (INV-13) | ζ is a parallel leaf; γ renders the state from `rule_vintage` and the absence of any row carrying the field; κ verifies the state before ζ lands and the value after | PASS (built + bound by γ) |
| Rows exist to read | DAG-direction | β upstream of γ | PASS |

## δ — Markers: derived stream, curated schema extension, one reader, ordinals  *(intermediate → ε)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `routing_decision` payload has `role, model, effort, budget_usd, max_turns, source_layer` | field-population (wired) | `orchestrator/src/orchestrator/routing_dispatch.py` emit; 24,841 df rows since 2026-07-16 (S3 §1) | PASS |
| `config_reload` payload has `applied {path: {old, new}}` | field-population | S3 §3 (df 26 rows); `escalation` MCP `reload_config` writer | PASS |
| `runs.code_sha` populated | field-population | `orchestrator/src/orchestrator/harness.py` start_run `deployed_sha`; non-null from 2026-10-21 only — the rug is labelled from that date, never coalesced | PASS |
| Curated file + fail-soft loader to extend | capability→producer (wired) | `dashboard/regime-markers.yaml`; `dashboard/src/dashboard/data/escalation_analytics.py::load_regime_markers` (ignores unknown keys) | PASS |
| Landing axis to resolve ordinals against | DAG-direction | γ upstream of δ | PASS |

## ε — Quality tab and chart primitives  *(intermediate → ι, κ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Hand-rolled SVG chart with a signature to extend | capability→producer (wired) | `dashboard/src/dashboard/static/redux/charts.jsx::LineChart({series, labels, height, yLabel, formatY, formatX, countAxis})` — no band/marker prop today | PASS |
| `StatTile({label, datum, history})` | capability→producer (wired) | one-datum PRD γ1 (task 5588 done) | PASS |
| Regime-marker overlay to lift | capability→producer (wired) | `static/redux/tab_escalation_analytics.jsx::RegimeMarkers`, `::TimeChart` | PASS |
| Tab/endpoint wiring sites | capability→producer (wired) | `static/redux/{app.jsx, poll_scope.js, data.js, window_chip.js, index.html}`; pins `dashboard/tests/test_index_html.py`, `test_cache_buster_freshness.py::_REQUIRED_ASSETS`, `tests/js/classic_script_scope.test.mjs` | PASS |
| Payload to render | DAG-direction | δ (and γ) upstream of ε | PASS |

## ζ — Orchestrator: typed `charter_class` and `prompt_digest`  *(intermediate → κ; parallel to α–ε)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| The four architect block sites exist and write `block_reason` | capability→producer (wired) | `orchestrator/src/orchestrator/workflow.py` strings `Architect reported task already done …`, `Architect reported task unactionable`, `Architect reported false RED-test premise`, `Plan files not touched by branch` | PASS |
| `charter_class` does not collide with the unblock domain | rejection-mechanism (naming) | `orchestrator/src/orchestrator/unblock_types.py::BlockClass` is the closed 4-value unblock domain (`classify_block_reason`); the new field is a separate name on an orthogonal dimension; `charter_class` has 0 hits on main | PASS |
| `task_results` additive column pattern | capability→producer (wired) | `event_store.py` task_results DDL + the `_ADDED_*` migration idiom (one-datum PRD burndown precedent: nullable, never `DEFAULT 0`) | PASS |
| `routing_decision` emit site to extend with `prompt_digest` | capability→producer (wired) | `routing_dispatch.py` routing_decision emit; role prompt constants in `agents/roles.py`; `prompt_digest` has 0 hits on main | PASS |

## η — Snapshot identity: schema 3, identity test, labelled diff, cadence wrapper  *(intermediate → θ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Instrument + params header to extend | capability→producer (wired) | `scripts/quality_metrics_snapshot.py::SCHEMA_VERSION` (= 2), `::diff_lines`, `params` carrying `complexipy_version`; `scripts/source_measures.py::COMPLEXIPY_MIN` | PASS |
| Byte-identity at one identity (the premise of decision 17's test) | premise (G6 branch 2) | measured: runs B/C/D at `922d9d3baa` byte-identical below `run_id`/`cost` across path, locale, CPython 3.13.9/3.14.4 (S5 §1) | PASS |
| Schema-1/2 "unknown" polarity to reuse for schema 3 | capability→producer (wired) | `tests/scripts/test_quality_metrics_snapshot.py::test_a_schema_1_snapshot_names_its_cycle_sets_unknown` | PASS |
| Pinned-tree recipe | capability→producer | `skills/review-all/SKILL.md` Phase 0 (`git worktree add --detach`); S5 `--shared` clone probe | PASS |
| The exact-identity skip is permitted | rejection-mechanism (policy) | PRD decision 18 argues it against `plans/quality-metrics-snapshot-prd.md` decision 5 and merge-lane-quality Correction 7; Leo ruled 2026-11-04 | PASS |

## θ — Nightly timer, weekly backfill, `/review-all` previous-snapshot rule  *(intermediate → ι)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Timer template precedent | capability→producer (wired) | `scripts/legibility-trickle@.timer` / `.service` (OnCalendar, Persistent, instance per project) | PASS |
| Phase 1 previous-snapshot rule to extend | capability→producer (wired) | `skills/review-all/SKILL.md` Phase 1 step 1 ("Previous snapshot") | PASS |
| `data/` is gitignored | capability→producer | `.gitignore` `/data/` | PASS |
| Wrapper exists | DAG-direction | η upstream of θ | PASS |

## ι — Snapshot step-series datum and tab section  *(intermediate → κ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Per-member summary figures defined by the instrument | capability→producer (wired) | `scripts/quality_metrics_snapshot.py` `--summary` (`summary_table`) | PASS |
| Committed baselines to read | field-population | `plans/quality-metrics/review-all-dark_factory-20261008.json` (schema 1, reads as "instrument unknown") | PASS |
| Nightly snapshots to read | DAG-direction | θ upstream of ι; until the first nightly run the datum renders "no snapshot yet" (INV-13) | PASS |
| Endpoint and tab to extend | DAG-direction | γ, ε upstream of ι | PASS |

## κ — Integration gate  *(leaf; the PRD's signal leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Boundary-test sketch rows 1–21 executable against live df/reify stores | DAG-direction | all producers upstream (ε, ζ, ι and their chains) | PASS |
| Rows 14–15 need a dispatched run | premise (G6 branch 3) | the gate observes a real architect block and a real routing decision on main after ζ lands; if none occurs in its window it records "not yet observed", never a seeded row (INV-13) | PASS |
