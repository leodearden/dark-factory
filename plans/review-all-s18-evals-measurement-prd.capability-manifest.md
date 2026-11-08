# Capability manifest — evals measurement integrity (review-all S18)

Binds each leaf's asserted capabilities to evidence, mechanizing G3 + G6.
Machine-readable twin: `review-all-s18-evals-measurement-prd.capability-manifest.yaml`.
PRD: `plans/review-all-s18-evals-measurement-prd.md`.
As of main `ae3047ec3a`, 2026-10-10.

**Verdict summary: 53 bindings across 11 leaves, all PASS; 46 mechanical checks, every one
evaluated FAIL at `main` (the undelivered state, as required) with
`shared.delivered_check_polarity.evaluate_grep_at_tree` / `evaluate_path_at_tree`; zero
polarity-lint findings from `lint_delivered_checks`. At decompose (2026-10-10) the lint was re-run
with each leaf's `metadata.files`; it flagged γ's `no-fabricated-tie-or-zero` as
`absent_overbroad` (the phrase also appears in the committed data files `evals/judge_report.json`
and `evals/judge_state.json`), so that check's `paths` were narrowed to `evals/judge.py`,
`evals/compare.py` and `evals/reviewer_trial/scorer.py`, after which every leaf linted clean.
Stamped by `commit_planning`: α 6733, β 6734, γ 6736, δ 6737, ε 6738, ζ 6739, η 6741, θ 6743,
ι 6746, κ 6748, λ 6752. 7 manual bindings (boundary tests B1–B5, the report
snapshot, the implementer-survivor measurement).**

No binding is declared-only, test-only, producer-downstream or producer-absent inside the batch.
One producer is outside it: λ's `evals/paths.py` consumes S14's `shared/project_layout.py`, absent
on main, and decompose wires λ to S14's filed leaf (PRD §9). If S14 has not filed when this batch
is decomposed, λ is filed with that dependency recorded in its description and the edge is added
when S14 files.

## Substrate findings that shaped the bindings

| Capability | Status (re-verified at `ae3047ec3a`) | Consequence |
|---|---|---|
| Transport-refusal classifier | **CONFIRMED** — `shared/src/shared/invocation_outcome.py::classify_invocation(result, *, strict_confirm, backend)`; variants `CapHit`, `AuthFailed`, `ModelNotFound`, `ServerError`, `ZeroOutputWedge`, `OK` | β's `classify_unmeasured` delegates to it (S20 owns) |
| Today's eval wrapper | **CONFIRMED** — `evals/metrics.py::detect_invocation_error` returns six string prefixes; called only from `runner.run_architect_eval` and `judge.judge_plan_quality` | β replaces it; prefixes become `UnmeasuredCause` values |
| Fabricating sites | **CONFIRMED** — `reviewer_trial/scorer.py::match_issues` returns `([], reviewer_issues, cost)`; `judge.py::run_judge` and `compare.py::_assess_task` fall back to `'tie'`; `compare_models`' synthesizer falls back to `overall_winner='tie'`; `elo.py::record_match` maps an unknown winner to 0.5 | γ's scope |
| Gate retry wrapper | **CONFIRMED** — `shared.cli_invoke.invoke_with_cap_retry(gate, label, *, cap_wait_sanity_secs, invoke_fn, backend, **kwargs)` | `measured_call` uses it when given a gate |
| Results-dir re-derivations | **CONFIRMED** — `RESULTS_DIR` at module level in `evals/runner.py`, `evals/rereview.py`, `scripts/run_vllm_eval.py`, `orchestrator/scripts/backfill_composite_scores.py`; filename glob in `runner._result_exists` and twice in `run_vllm_eval.py`; `split('__')` in `rereview._extract_run_id` | α's absent checks |
| Packaged cells | **MEASURED** — 686 cells parse; `role_under_test` is `None` on 335, `architect` on 335, `judge` on 16; 3 carry a string `invocation_error` (`architect:cap_exhausted: …`) | D3 legacy rules; ε's survivor measurement |
| Implementer survivors today | **MEASURED** — `select_survivors(build_composite_report(cells), top_k=3, roles=['implementer'])` returns an empty implementer list; `roles=['architect']` returns three | ε's signal is falsifiable |
| Verdict envelope | **CONFIRMED** — `orchestrator/mcp/verdict_tools.py::_envelope` (`schema_version`, `role`, nested `verdict`); `artifacts.py::TaskArtifacts.aggregate_reviews` applies the ERROR rule; six ERROR literal comparisons across `evals/` and `artifacts.py` | ζ's scope |
| Plan predicate reads | **CONFIRMED** — `judge.is_scorable_plan` reads `steps`; `judge._plan_confirmed` reads `_finalized_at` (S9's marker key) | β moves them unchanged; S9 migrates later |
| Gate builder in tests | **CONFIRMED** — `_build_eval_usage_gate` named in `test_eval_architect.py`, `test_eval_driver.py`, `_campaign_gate_helpers.py` | δ's absent check |
| Eval CLI | **CONFIRMED** — `cli.py` commands `eval`, `eval-list-fixtures`, `eval-sample`, `eval-ofat`, `eval-matrix`, `eval-confirm`; no command renders a report from existing cells (`_emit_composite_report` runs only after a stage) | PRD §11 "report snapshot" procedure instead of a CLI signal |
| Routing resolver | **CONFIRMED** — `orchestrator/routing.py::resolve_route(inputs: RouteInputs, config: OrchestratorConfig)` | κ routes `_run_plan_only` through it |
| Checkout literals | **CONFIRMED** — 14 quoted `/home/leo/…` literals in `orchestrator/src/orchestrator/evals/**/*.py` | λ's absent check |
| Shared project layout | **ABSENT** — no `shared/src/shared/project_layout.py` on main | λ depends on S14's leaf |
| Rationale home | **CONFIRMED** — `docs/plan-scoring-and-judge.md` exists | δ and ι move prose there |

## Bindings

### α — ResultStore owns the persisted eval cell

| Capability | Binding | Verdict |
|---|---|---|
| `result-store-class` | producer — `ResultStore` in `evals/results.py` (C2) | PASS |
| `eval-result-home` | producer — `EvalResult` moves to `evals/results.py` | PASS |
| `no-second-results-dir` | SPOT — four module-level `RESULTS_DIR` outside results.py removed | PASS |
| `no-second-filename-grammar` | heuristic 12 — glob and `split('__')` re-derivations removed | PASS |
| `loud-load-two-way` | manual — B1 | PASS |

### β — Typed measurement outcome; plan predicates leaf

| Capability | Binding | Verdict |
|---|---|---|
| `unmeasured-cause-enum` | producer — `UnmeasuredCause` in `evals/measurement.py` | PASS |
| `measured-call-helper` | producer — `measured_call`; consumers γ (in batch, downstream) | PASS |
| `classifier-on-shared-seam` | G4 — `classify_unmeasured` on S20's `classify_invocation` (present) | PASS |
| `no-string-marker-producer-or-parser` | heuristic 12 — `detect_invocation_error` and `_taint_cause` removed | PASS |
| `plan-predicates-leaf` | producer — `is_scorable_plan` in `evals/plan_predicates.py` | PASS |
| `no-metrics-judge-local-imports` | weakened key re-verified — the pair's local imports removed | PASS |
| `legacy-and-new-cells-share-buckets` | manual — B3 + report snapshot | PASS |

### γ — Every LLM scoring site measures or says it did not

| Capability | Binding | Verdict |
|---|---|---|
| `no-neutral-fallback-parse` | idiom absent from judge, compare, reviewer_trial/scorer | PASS |
| `no-fabricated-tie-or-zero` | fabricated fallbacks removed | PASS |
| `scoring-result-unmeasured-state` | producer — `ScoringResult.unmeasured` | PASS |
| `elo-unmeasured-bound` | producer — `MAX_UNMEASURED_PER_PAIR` in `elo.py` | PASS |
| `reducers-exclude-and-count` | manual — B2; producer β upstream | PASS |

### δ — `run_architect_eval` as invoke → classify → assemble; one gate lifecycle

| Capability | Binding | Verdict |
|---|---|---|
| `architect-attempt-type` | producer — `ArchitectAttempt` in `evals/architect_eval.py` | PASS |
| `pure-classify-step` | producer — `classify_attempt` | PASS |
| `owned-gate-context-manager` | producer — `owned_usage_gate` in `evals/eval_gate.py`; consumer ε downstream | PASS |
| `tests-stop-patching-private-gate-builder` | tests stance — `_build_eval_usage_gate` absent from tests | PASS |
| `no-function-local-collaborator-imports` | five local imports become injected deps | PASS |
| `rationale-to-docs` | manual — prose relocation | PASS |

### ε — One workflow-cell executor; `RoleUnderTest`

| Capability | Binding | Verdict |
|---|---|---|
| `role-under-test-enum` | producer — `RoleUnderTest` in `evals/metrics.py` | PASS |
| `no-role-dict-mutation` | dict stamping removed | PASS |
| `one-cell-executor` | producer — `_run_workflow_cell` | PASS |
| `no-hand-rolled-gate-teardown` | three teardown copies removed; producer δ upstream | PASS |
| `metrics-is-a-leaf` | `collect_metrics` leaves `metrics.py` | PASS |
| `implementer-survivors-exist` | manual — measured empty today, non-empty after | PASS |

### ζ — One review-payload normaliser

| Capability | Binding | Verdict |
|---|---|---|
| `normalize-review` | producer — `orchestrator/review_payload.py::normalize_review` | PASS |
| `aggregate-reviews-uses-normaliser` | production consumer wired in the same leaf | PASS |
| `no-error-literal-filters` | six ERROR literal comparisons removed | PASS |
| `no-eval-envelope-unwrapper` | `_unwrap_verdict_envelope` removed | PASS |
| `production-eval-parity` | manual — B4 | PASS |

### η — Reviewer-trial drivers

| Capability | Binding | Verdict |
|---|---|---|
| `no-second-panel-cache` | `_load_or_run_panel` and its "mirrors" note removed | PASS |
| `panel-result-from-dict` | producer — `PanelRunResult.from_dict` | PASS |
| `run-and-score-helper` | producer — `run_and_score` | PASS |

### θ — `report.py` leaves

| Capability | Binding | Verdict |
|---|---|---|
| `stats-module` | producer — `evals/stats.py::mean_ci95` | PASS |
| `prices-module` | producer — `evals/prices.py::build_price_table` | PASS |
| `elo-report-module` | producer — `evals/elo_report.py::compute_tiers` | PASS |
| `cell-admission-module` | path — `evals/cell_admission.py` | PASS |
| `table-render-module` | path — `evals/table_render.py` | PASS |

### ι — `report.py` surfaces

| Capability | Binding | Verdict |
|---|---|---|
| `report-module-gone` | path absent — `evals/report.py`, no shim | PASS |
| `composite-report-module` | producer — `evals/composite_report.py::build_composite_report` | PASS |
| `plan-quality-report-module` | producer — `evals/plan_quality_report.py::build_plan_quality_report` | PASS |
| `snapshot-unchanged` | manual — report snapshot | PASS |

### κ — Eval CLI group

| Capability | Binding | Verdict |
|---|---|---|
| `eval-cli-module` | producer — `evals/cli.py::eval_cmd` | PASS |
| `no-eval-drivers-in-cli` | `_run_matrix_cmd`, `_run_plan_only`, `_run_judge_cmd` leave `cli.py` | PASS |
| `evals-never-import-cli` | upward import removed | PASS |
| `plan-only-routed` | `resolve_route(` in `evals/cli.py`; substrate present | PASS |
| `config-preamble-decorator` | path — `orchestrator/cli_config.py` | PASS |

### λ — One checkout root

| Capability | Binding | Verdict |
|---|---|---|
| `no-checkout-literals` | 14 quoted literals removed | PASS |
| `paths-module` | path — `evals/paths.py`; upstream producer S14 (cross-stream dependency) | PASS |
