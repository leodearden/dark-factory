# PRD — Evals measurement integrity (review-all stream S18)

Status: **active**, authored 2026-10-10 by the S18 `/prd` seat of the `/review-all` remediation
program (`plans/review-all-program-dark_factory-2026-10-10.md`, wave 1, agent mode). Approved as
proposed under ruling R21. Finding run `review-all-dark_factory-20261008`, pinned at `6551b75aba`.

**Code anchors** are cited by symbol. Every anchor below was re-read on main `ae3047ec3a`
(2026-10-10); `git diff 6551b75aba..main` touches none of this stream's files except
`orchestrator/src/orchestrator/artifacts.py` (task 4618's claimant-TTL binding, outside
`TaskArtifacts.aggregate_reviews`).

## 1. Goal

An eval number is either a measurement or it says, in the result a caller holds, that it is
not one. After this PRD:

- every LLM call whose answer becomes a score goes through one eval-layer helper that returns
  `Scored(value)` or `Unmeasured(stage, cause, reason)`, and every reducer that averages or
  ranks those results excludes the unmeasured ones and reports how many it excluded;
- the cause of an unmeasured cell is a typed value end to end, from the classifier to the
  persisted cell to the report bucket, with no string grammar in between;
- the persisted-cell contract (directory, filename grammar, `EvalResult`, load normalisation)
  has one owner, `evals/results.py::ResultStore`;
- the most-edited eval code (`run_architect_eval`, the workflow-cell executors, `report.py`,
  the eval half of `cli.py`) is partitioned so the next agent change loads one concern.

The user-observable surface is the eval CLI and its reports (`orchestrator eval` with its
`--judge`/`--compare`/`--report` modes, `eval-ofat`/`eval-matrix`/`eval-confirm`, `python -m
orchestrator.evals.reviewer_trial campaign`, the judge/compare commands) and the persisted cells
under `orchestrator/src/orchestrator/evals/results/`. Consumers: the operator reading OFAT and
campaign reports to decide routing changes, and the scripts drivers
(`scripts/run_fable_trial_v2_campaign.py`, `scripts/run_judge_ofat_pilot.py`,
`scripts/run_vllm_eval.py`).

## 2. Background

The review's area summary for `orchestrator/evals`: the pure leaves (prompt_opt engine,
reviewer_trial corpus, snapshots, `metrics.blend_composite`) are healthy; the cost sits in
`runner.py` (2,533 lines) and `report.py` (2,232 lines), and in persisted cells that are
`dict[str, Any]` read by string keys and string literals. Task 3118's rule — a transport
refusal is not a content measurement — is enforced only on the architect path. Three other
scoring sites turn a refusal into a scored tie or a zero.

Prior work this builds on: task 3118 (`detect_invocation_error`), 3302/3303 (`is_scorable_plan`
anti-fabrication floor), 2478 (OFAT driver), 2524 (verdict-envelope readers), 4844
(`replay_frame.py`). The rationale those tasks wrote into docstrings moves to
`docs/plan-scoring-and-judge.md`, which already exists; code keeps a pointer.

## 3. Finding keys

| Key | Sev | Verdict | Anchor (symbol) | Closed by |
|---|---|---|---|---|
| `fk-0d4c1014e41f` | high | confirmed | `evals/reviewer_trial/scorer.py::match_issues` | γ |
| `fk-3505aba0481d` | high | confirmed | `evals/runner.py::run_architect_eval` | δ |
| `fk-43584c88970d` | medium | confirmed | `evals/runner.py::save_result` | α |
| `fk-b6d389a4fcf2` | medium | confirmed | `evals/report.py::_taint_cause` | β |
| `fk-28ca5c64de06` | medium | confirmed | `evals/metrics.py::EvalMetrics` | ε |
| `fk-b80c8e52e526` | medium | confirmed | `evals/runner.py::run_end_to_end` | ε |
| `fk-2e8daab1913f` | medium | confirmed | `evals/scoring.py::_unwrap_verdict_envelope` | ζ |
| `fk-294ee8dfa70f` | low | confirmed | `evals/reviewer_trial/__main__.py::_load_or_run_panel` | η |
| `fk-df50bc042300` | low | **weakened** | `evals/metrics.py::resolve_terminal_kind` | β (pair) + ε (collect_metrics) |
| `fk-b2c28b4f8f17` | low | confirmed | `evals/report.py` | θ, ι |
| `fk-4352b1ee49f5` | low | confirmed | `orchestrator/cli.py::eval_cmd` | κ |
| `fk-14926fe151d4` | low | confirmed | `slug:evals-hardcoded-checkout-paths` | λ |

Related, not this stream's key: `fk-1529b339d977` (`evals/prompt_opt/__init__.py` re-exports)
was filed by the run as mechanical task **6597** (pending). It stays standing; λ, which edits
`prompt_opt` modules, depends on it so the new paths module is never added to a facade that
6597 deletes.

Pre-existing task subsumed: **6324** ("Measure/partition evals/runner.py", pending, low). δ
and ε remove `run_architect_eval`'s body and the duplicated executor code from `runner.py`,
which is what 6324 asks to decide. Decompose cancels 6324 with `x_acceptance_reason` naming δ
and ε.

**Weakened key re-verified (R23, G3/G6).** `fk-df50bc042300` at `ae3047ec3a`: the deferred
imports `metrics.resolve_terminal_kind → judge.is_scorable_plan`, `judge.judge_plan_quality →
metrics.{coerce_cost_usd, detect_invocation_error}`, `metrics.collect_metrics →
scoring.recovery_score_for_run` and `scoring.recovery_score_for_run → snapshots` are all
present, and the two comments still contradict each other about a cycle. The skeptic's
reduction stands: hoisting would not raise `ImportError` today. The residual cost is four
deferred imports and two contradictory comments, which β and ε remove as a by-product of
moves they make anyway. Kept at low; not dropped.

## 4. Sketch of approach

Seven new modules, each with a one-sentence purpose (heuristic 6), each importing only
downward:

| Module | Purpose |
|---|---|
| `evals/results.py` | Owns the persisted eval cell: `EvalResult`, `ResultStore(dir)`, the filename grammar, and load-time normalisation of legacy cells. |
| `evals/measurement.py` | Classifies an LLM invocation as scored or unmeasured and names why, as typed values. |
| `evals/plan_predicates.py` | Answers "does this plan artifact / persisted cell carry a plan?" (`is_scorable_plan`, `produced_a_plan`). |
| `evals/architect_eval.py` | The architect cell's invoke, classify and assemble steps, each with injected collaborators. |
| `evals/eval_gate.py` | The eval-side `UsageGate` lifecycle: build, inject, own, shut down. |
| `orchestrator/review_payload.py` | Normalises a reviewer artifact (verdict-tool envelope or legacy payload) to a typed `ReviewPayload`. |
| `evals/paths.py` | Resolves every eval default path from one checkout root. |

Plus the partition of `report.py` (θ, ι) and of `cli.py`'s eval section into `evals/cli.py`
(κ).

## 5. Contracts (approach B + H)

G5 fires: γ's reducers and the scripts drivers consume α's and β's types across leaf
boundaries, and ζ's normaliser is a production↔eval seam. Each contract is fixed here; the
owning leaf lands it with a two-way boundary test (§6).

**C1 — measurement outcome (`evals/measurement.py`, β).**
- `MeasurementStage(StrEnum)`: `architect`, `judge` (the plan judge), `pairwise_judge`,
  `matcher`, `assessor`, `synthesizer`.
- `UnmeasuredCause(StrEnum)`: the six transport causes `detect_invocation_error` names today
  from `classify_invocation` (`cap_hit`, `auth_failed`, `model_not_found`, `api_error` — the
  body-less 429 — `server_error`, `wedge`), the four harness causes `run_architect_eval` mints
  today (`timeout`, `cap_exhausted`, `harness_error`, `raised` — the plan judge raised),
  `unparseable` (the instrument answered but its answer could not be parsed), and
  `legacy_unclassified` (load-time only, §7 D3). Values equal today's marker prefixes, so the
  legacy mapping is identity and `cap_excluded_by_cause` bucket labels do not move. The
  plan-judge stage's value is `judge`, today's stage prefix.
- `Unmeasured(stage, cause, reason: str)` frozen; `Scored[T](value: T, cost_usd: float)`
  frozen; `Unmeasured` also carries `cost_usd` (a refused call still spends).
- `classify_unmeasured(result, *, stage, backend) -> Unmeasured | None` replaces
  `metrics.detect_invocation_error`. It is built on
  `shared.invocation_outcome.classify_invocation(result, strict_confirm=True, backend=…)` and
  keeps today's precedence (CapHit/AuthFailed/ModelNotFound, OK short-circuit, body-less 429,
  ServerError, ZeroOutputWedge). It never re-matches cap strings.
- `async measured_call(invoke, parse, *, stage, backend, usage_gate=None) -> Scored[T] |
  Unmeasured`. `invoke` is a zero-argument awaitable factory returning `AgentResult`; with a
  gate it runs through `shared.cli_invoke.invoke_with_cap_retry`. `parse(result) -> T` raises
  on an unparseable answer; `measured_call` turns that into `Unmeasured(cause=unparseable)`.
  It never fabricates a value.
- Persisted form: `EvalMetrics.invocation_errors: list[{stage, cause, reason}] | None`
  (renamed from the string `invocation_error`; plural because the architect cell can carry an
  architect and a judge record). Rendering to text happens only in report formatters, which
  render the records as today's `{stage}:{cause}: {reason}` joined by `; `, so the
  plan-quality table's `invocation_error` column reads as it does now.

**C2 — persisted cell (`evals/results.py`, α).**
- `EvalResult` moves here unchanged in fields.
- `ResultStore(dir)`: `save(result) -> Path`, `load_all() -> list[EvalResult]`,
  `exists(task_id, config_name) -> bool`, `path_for(result)`, `parse_name(path) ->
  CellName(task_id, config_name, run_id)`. `ResultStore.packaged()` is the default
  `evals/results/` store. A malformed file raises `ResultLoadError` naming the file (today
  only the scripts copy does). A missing dir on `load_all` raises naming the dir.
- Load-time normalisation is one function, `normalize_legacy_metrics(metrics) -> metrics`,
  applied by `load_all`; β and ε each add their legacy rule to it (§7 D3).

**C3 — review payload (`orchestrator/review_payload.py`, ζ).**
- `Verdict(StrEnum)`: `PASS`, `ISSUES_FOUND`, `ERROR`.
- `ReviewPayload` frozen: `verdict: Verdict`, `issues: tuple[dict, ...]`, `reviewer: str |
  None`. `normalize_review(artifact: dict) -> ReviewPayload` unwraps the verdict-tool envelope
  (`mcp/verdict_tools.py::_envelope` shape) or takes the legacy payload, and applies the ERROR
  rule once: an ERROR payload has `issues == ()`.
- `TaskArtifacts.aggregate_reviews` is rewritten over `normalize_review`, so production and the
  eval readers share one reading.

**C4 — architect cell steps (`evals/architect_eval.py`, δ).**
- `ArchitectAttempt` frozen: plan, declines, token counts, turns, duration, cost, `outcome`,
  `unmeasured: Unmeasured | None`.
- `async invoke_architect(task, config, deps) -> ArchitectAttempt` — the only step with
  effects; the decline read stays before worktree cleanup inside it.
- `classify_attempt(attempt) -> PlanScoring` pure — the taint table: `tainted` (unmeasured and
  no scorable plan), `floored` (unmeasured with a plan, or no scorable plan), or `judge`.
- `assemble_metrics(attempt, scoring, judge_outcome, cost) -> EvalMetrics` pure.
- `ArchitectEvalDeps` frozen: `create_worktree`, `cleanup_worktree`, `invoke`, `judge`,
  `gate_factory`, `store`. `runner.run_architect_eval` keeps its public signature and becomes
  a short composition of the three steps over default deps.

## 6. Boundary-test sketch

| # | Seam | Two-way test |
|---|---|---|
| B1 | executor ↔ scripts (C2) | A cell saved by `run_eval`'s save path into a tmp `ResultStore` loads through `scripts/_eval_results_io.load_results_from_dir` and through `ResultStore.load_all` to equal `EvalResult`s; a truncated file in the same dir fails both with the file named. |
| B2 | classifier ↔ every scoring site (C1) | One real-shaped refused `AgentResult` (the task-3118 CLI shape: `is_error`, `api_error_status=429`, cap-message body, `cost_usd>0`) driven through the matcher, pairwise judge, assessor, synthesizer and plan judge yields `Unmeasured` at each, and the reducers above them (`summarize_campaign`, `build_trial_report`, the Elo tournament, `compare_models`) report it in their unmeasured count and leave every mean unchanged. |
| B3 | producer ↔ report (C1, C2) | A legacy on-disk cell with `invocation_error: "architect:cap_exhausted: …"` and one written by the new producer both land in the same `cap_excluded_by_cause` bucket of `build_plan_quality_report`. |
| B4 | production writer ↔ eval readers (C3) | An envelope produced by `verdict_tools._envelope` for each `Verdict`, and a legacy payload, give identical blocking/suggestion counts through `TaskArtifacts.aggregate_reviews` and through `scoring.quality_from_review_artifact` / `compare.score_result_from_artifacts`. |
| B5 | architect steps (C4) | The taint table is tested on `classify_attempt` alone, one case per exit; one end-to-end test through `run_architect_eval` with injected deps asserts the decline read precedes cleanup. |

## 7. Resolved design decisions

- **D1 — An unparseable instrument answer is unmeasured, not a tie or a zero.** The matcher,
  pairwise judge, assessor and synthesizer are instruments, not the subject under test; their
  parse failure is a failed measurement whatever its cause. The subject's own content failure
  (an architect that ran and produced a bad plan) stays a scored signal, as 3118/3302 ruled.
  This is why the cause set has `unparseable`, and why `test_reviewer_trial_scorer.py::
  test_unparseable_output_still_reports_cost` is rewritten to assert `Unmeasured` with the
  real cost.
- **D2 — Reducers exclude and count.** Each reducer gains an `unmeasured` count beside the
  existing denominator; a group with zero measured records reports `None`, never `0.0`. The
  Elo tournament does not call `record_match` for an unmeasured verdict and does not count it
  toward `MAX_PER_PAIR`; it counts unmeasured attempts per pair against a separate bound
  (`MAX_UNMEASURED_PER_PAIR`, default 2) so a capped account cannot spin the loop. The
  reviewer-trial campaign never writes an unmeasured score to its score cache, so a rerun
  re-measures it.
- **D3 — Legacy cells are normalised once, at load.** `ResultStore.load_all` maps a string
  `invocation_error` to one `invocation_errors` record (the stage and cause read by the
  rule `report._taint_cause` applies today, moved, not copied; an unrecognised prefix becomes
  `legacy_unclassified`), and an absent `role_under_test` to `implementer` (ε; every
  `None`-role cell on disk was written by `run_eval` without a judge config — 335 of 686 at
  `ae3047ec3a`). After α/β/ε, `_taint_cause` and every literal role comparison are gone. INV-13
  backfill decision per record kind: eval cells — migrated at load; reviewer-trial cached
  score files written before γ — abandoned (no marker exists to recover; documented in γ).
- **D4 — `role_under_test` is one vocabulary.** `RoleUnderTest(StrEnum)`: `implementer`,
  `architect`, `judge`, `end_to_end`, in `evals/metrics.py`. `EvalConfig.role` is typed
  `RoleUnderTest` and validated to the first three (`end_to_end` is a methodology stage, not a
  candidate role). `collect_metrics(…, role=)` sets it on the dataclass; nothing mutates
  `metrics_dict`.
- **D5 — `run_architect_eval` keeps its name and signature.** Its parts move to
  `evals/architect_eval.py`; callers and the stage drivers are unchanged. This avoids a
  re-export shim (heuristic 13) and avoids touching ~20 importers for no gain.
- **D6 — The gate lifecycle has one home.** `evals/eval_gate.py` owns `InjectedGate`,
  `build_eval_usage_gate` (public, was `_build_eval_usage_gate`), `owned_usage_gate(injected,
  orch_config, factory=build_eval_usage_gate)` (async context manager) and
  `campaign_usage_gate`. All three executors use `owned_usage_gate`; tests inject a factory
  instead of patching a private name.
- **D7 — One workflow-cell executor.** `_run_workflow_cell(task, orch_config, *,
  initial_plan, briefing, label, role, usage_gate, store)` carries the 76 shared lines;
  `run_eval` and `run_end_to_end` become thin callers; `run_eval_matrix` becomes thunks over
  `_bounded_fanout` with the skip-existing check inside the thunk. `collect_metrics` moves from
  `metrics.py` into the executor module, leaving `metrics.py` a leaf of vocabularies and pure
  functions.
- **D8 — The normaliser is orchestrator code, consumed by evals.** `orchestrator/
  review_payload.py`, a new module rather than an addition to `artifacts.py`, so it does not
  enlarge a file S9 restructures. `aggregate_reviews` and the five eval sites call it.
  `workflow._run_reviewer`'s own unwrap is not migrated here (§10).
- **D9 — `report.py` is split along its measured partition, not by size.** Leaf modules first
  (θ: `stats.py`, `prices.py`, `cell_admission.py`, `table_render.py`, `elo_report.py`), then
  the two report surfaces (ι: `composite_report.py` with methodology and `select_survivors`;
  `plan_quality_report.py` with recovery). `report.py` is deleted; no re-export shim. Per-task
  rationale moves to `docs/plan-scoring-and-judge.md`. Heuristic 14's measurement protocol is
  satisfied by the review's partition evidence (no upward import or cycle among the parts).
- **D10 — The eval CLI is a click group in `evals/cli.py`** registered by `cli.py` with
  `main.add_command` (downward import). `_run_plan_only` resolves its model through
  `orchestrator.routing.resolve_route` with `role_name='architect'` and the eval config. The
  repeated `load_config`/`ConfigRequiredError` preamble becomes one decorator in
  `orchestrator/cli_config.py`, applied to every command that has it, eval or not.
- **D11 — One checkout root.** `evals/paths.py` derives every default from
  `shared.project_layout` (S14) for the main checkout and known project roots; the six
  per-path env vars collapse to S14's root override. A path S14's leaf does not expose is
  requested on S14's leaf, not spelled here.

## 8. Pre-conditions

- `shared.invocation_outcome.classify_invocation`, `shared.cli_invoke.invoke_with_cap_retry`,
  `orchestrator.routing.resolve_route(RouteInputs, OrchestratorConfig)`,
  `mcp/verdict_tools.py::_envelope` — all present on main (G3, verified).
- `shared/project_layout.py` (S14) — **absent on main**; λ depends on S14's leaf (§9).
- Task 6597 (pending) — λ depends on it.

## 9. Cross-PRD relationship and seams (G4)

| Seam | Owner | This PRD's role |
|---|---|---|
| `evals/results.py ResultStore` + `EvalResult` | **S18 (α)** | provides; `scripts/_eval_results_io.py`, `orchestrator/scripts/backfill_composite_scores.py`, `scripts/run_vllm_eval.py`, `scripts/run_judge_ofat_pilot.py` import it |
| `orchestrator/review_payload.py` normaliser | **S18 (ζ)**, orchestrator code | provides; `artifacts.aggregate_reviews` and the eval readers consume. Later adopters (`workflow._run_reviewer`, `review_suggestions/prior_round.py`) are S9's / a follow-up's to migrate, never a second normaliser |
| `evals/measurement.py` (C1) | **S18 (β)** | provides; γ and δ consume |
| `shared.invocation_outcome.classify_invocation` | S20 (R18) | consumes the current signature; never forks the cause tables. If S20's leaf changes the signature, S20 updates `classify_unmeasured` as a caller |
| `shared.cli_invoke.invoke_with_cap_retry` | S20 | `measured_call` and `invoke_architect` call it; S20's state-machine rewrite updates callers |
| Plan model / `PlanState` in `artifacts.py` | S9 | `evals/plan_predicates.py` moves `is_scorable_plan` and `_plan_confirmed`'s marker read unchanged; S9's consumer migration rewrites them against `PlanState` |
| `shared.project_layout` locators | S14 | λ consumes; dependency on S14's project_layout leaf |
| `scripts/_checkout.py` bootstrap | S25 | α, ε and κ edit scripts that S25's bootstrap leaf also edits; same-file serialisation is wired at decompose against S25's filed leaves |
| `orchestrator.routing.resolve_route` | S16 | κ consumes it for `_run_plan_only`; the other literal-model eval invocations are S16's `fk-dac0a2a67201` |

No reciprocal "the other owns it" pattern remains: the brief's "normaliser is orchestrator
code" and the program table's "S18 owns it" are the same fact — S18 lands it, in the
orchestrator package, outside `evals/`.

## 10. Out of scope

- Migrating `workflow._run_reviewer` and `review_suggestions/prior_round.py` onto
  `normalize_review` (S9 restructures `workflow.py`; recorded as a consumer in §9).
- Routing the remaining eval `invoke_agent` sites through `resolve_route` (S16).
- Re-scoring reviewer-trial score files cached before γ (D3: abandoned).
- `mcp/verdict_tools.py` validating against `Verdict` (a later adopter).
- `prompt_opt/__init__.py` (task 6597).

## 11. Decomposition plan

Sizing per the overlay bands; move-dominated leaves are marked.

**Report snapshot.** No CLI command renders the composite or plan-quality report from existing
cells (`_emit_composite_report` runs only after `eval-ofat`/`eval-matrix`/`eval-confirm`), so
the leaves that must not move a reported number (α, β, ι) record a before/after snapshot:
`format_composite_table(build_composite_report(cells))` and
`format_plan_quality_table(build_plan_quality_report(cells))` over every cell in the packaged
results dir, run from the leaf's worktree against the main checkout's
`orchestrator/src/orchestrator/evals/results/` (the dir is gitignored, so worktrees hold no
cells). The two renders must be identical except where the leaf says a number changes and
why. Every leaf's brief cites
heuristic 13 (*files make internal sense in isolation*) and heuristic 14 (*no file too large —
no cheating by stitching files together with imports*) where it extracts a module.
Priority = the highest severity of the keys it closes.

**α — ResultStore owns the persisted eval cell** · medium · ~600 LOC · keys `fk-43584c88970d`
Files: `evals/results.py` (new), `evals/runner.py`, `evals/rereview.py`, `evals/compare.py`,
`evals/report.py`, `evals/judge_pilot.py`, `scripts/_eval_results_io.py`,
`scripts/run_vllm_eval.py`, `scripts/run_judge_ofat_pilot.py`,
`orchestrator/scripts/backfill_composite_scores.py`, tests
(`orchestrator/tests/test_eval_results_store.py` new, plus patch-target updates in
`test_runner_matrix.py`, `test_eval_driver.py`, `test_run_vllm_eval.py`,
`scripts/tests/test_run_judge_ofat_pilot.py`, `scripts/tests/test_run_fable_trial_v2_campaign.py`).
Crosses the 15-file review trigger only through test patch-target updates; coherent.
Signal: `git grep -nE '^RESULTS_DIR *=' -- '*.py'` matches only `evals/results.py`; `git grep -n
"split('__')" orchestrator/src/orchestrator/evals` is empty; B1 green; the report snapshot (below) over the packaged dir is unchanged (686
cells at `ae3047ec3a`).
Deps: none in batch; S25 same-file edges on the scripts.

**β — Typed measurement outcome on the architect path; plan predicates leaf** · medium ·
~800 LOC · keys `fk-b6d389a4fcf2`, `fk-df50bc042300`
Files: `evals/measurement.py` (new), `evals/plan_predicates.py` (new), `evals/metrics.py`,
`evals/judge.py` (`judge_plan_quality` via `measured_call`), `evals/runner.py` (architect
markers become `Unmeasured`), `evals/report.py` (bucket by `UnmeasuredCause`; `_taint_cause`
deleted), `evals/results.py` (legacy rule D3), `orchestrator/cli.py` (display of
`invocation_errors`), tests (`test_eval_measurement.py` new, `test_eval_architect.py`,
`test_eval_composite_report.py`, `scripts/tests/test_run_fable_trial_v2_campaign.py` fixtures).
Signal: a refused architect cell persists `invocation_errors: [{"stage": "architect",
"cause": "cap_hit", …}]`; `build_plan_quality_report`'s `cap_excluded_by_cause` keys are a
subset of `UnmeasuredCause` values; B3 green; the report snapshot is unchanged; `git grep -n "def detect_invocation_error\|def
_taint_cause" orchestrator/src` empty; no function-local import remains between `metrics`
and `judge`.
Consumes: S20's `classify_invocation`. Deps: α.

**γ — Every LLM scoring site measures or says it did not** · **high** · ~900 LOC · keys
`fk-0d4c1014e41f`
Files: `evals/reviewer_trial/scorer.py` (`match_issues`, `ScoringResult.unmeasured`),
`evals/reviewer_trial/campaign.py`, `evals/reviewer_trial/report.py`,
`evals/reviewer_trial/__main__.py` (score cache skips unmeasured; matcher gets the campaign
gate), `evals/judge.py` (`run_judge`, `run_elo_tournament`, `JudgeVerdict.unmeasured`),
`evals/elo.py` (unmeasured bound), `evals/compare.py` (`_assess_task`, synthesizer,
`TaskAssessment.unmeasured`, `ComparisonReport.unmeasured`), tests
(`test_reviewer_trial_scorer.py`, `test_reviewer_trial_campaign.py`,
`test_reviewer_trial_report.py`, `test_eval_diff_threading.py`, `test_eval_elo_unmeasured.py` new).
Signal: B2 green; `git grep -n "structured_output or json.loads("` over `evals/judge.py`,
`evals/compare.py` and `evals/reviewer_trial/scorer.py` is empty (the one parse helper lives in
`evals/measurement.py`); a reviewer-trial campaign over a fixture whose matcher is stubbed with the refused
result prints `unmeasured: 1` for that arm, its `mean_f1` equals the mean of the measured
records, and no score file is written for the refused cell.
Deps: β. Trigger-chain gate dependency (high).

**δ — `run_architect_eval` as invoke → classify → assemble; one eval gate lifecycle** ·
**high** · ~1,300 LOC (move-dominated) · keys `fk-3505aba0481d`
Files: `evals/architect_eval.py` (new), `evals/eval_gate.py` (new), `evals/runner.py`,
`evals/reviewer_trial/__main__.py` (import of `campaign_usage_gate`),
`docs/plan-scoring-and-judge.md` (taint-table and cap-wait rationale), tests
(`test_eval_architect.py`, `test_eval_driver.py`, `_campaign_gate_helpers.py`,
`test_reviewer_trial_campaign_gate.py`, `shared/tests/test_usage_gate_shutdown_pairing.py`
path string).
Signal: `run_architect_eval` is a composition of the three steps with no function-local
import; `git grep -n _build_eval_usage_gate orchestrator/tests` empty; the taint-table tests
call `classify_attempt` directly (B5); an `eval-ofat` architect run over one fixture writes a
cell byte-equivalent in its scoring fields to the pre-change cell for the same stubbed agent
result.
Deps: β (and γ for `reviewer_trial/__main__.py` same-file order). Trigger-chain gate dependency
(high).

**ε — One workflow-cell executor; `RoleUnderTest`** · medium · ~800 LOC · keys
`fk-b80c8e52e526`, `fk-28ca5c64de06`, `fk-df50bc042300`
Files: `evals/runner.py`, `evals/metrics.py` (`RoleUnderTest`; `collect_metrics` moves out),
`evals/configs.py`, `evals/report.py` (5 role readers), `evals/judge_pilot.py`,
`evals/results.py` (legacy role rule), `orchestrator/cli.py` (role reader),
`scripts/run_fable_trial_v2_campaign.py` (role reader), tests (`test_runner_matrix.py`,
`test_eval_driver_report.py`, `test_eval_driver.py`, `test_judge_ofat_pilot.py`).
Signal: a `run_eval` cell persists `role_under_test: "implementer"`;
`select_survivors(report, top_k=3, roles=['implementer'])` over `build_composite_report` of
the packaged results dir returns a non-empty `implementer` list (measured `{'implementer': []}`
over 686 cells at `ae3047ec3a`, while `roles=['architect']` returns three); `git grep -n
"metrics_dict\['role_under_test'\]\|owns_gate and usage_gate is not None"
orchestrator/src/orchestrator/evals` empty; `git grep -n "^async def collect_metrics"
orchestrator/src/orchestrator/evals/metrics.py` empty.
Deps: δ (same file), α.

**ζ — One review-payload normaliser** · medium · ~600 LOC · keys `fk-2e8daab1913f`
Files: `orchestrator/review_payload.py` (new), `orchestrator/artifacts.py`
(`aggregate_reviews` only), `evals/scoring.py`, `evals/compare.py`, `evals/rereview.py`,
`evals/reviewer_trial/runner.py`, `evals/reviewer_trial/scorer.py`,
`evals/reviewer_trial/__main__.py` (smoke check count), tests (`test_review_payload.py` new,
`test_eval_compare_review_artifacts.py`, `test_eval_driver_scoring.py`,
`test_reviewer_trial_runner.py`).
Signal: B4 green; `git grep -nE "[!=]= *['\"]ERROR['\"]" orchestrator/src/orchestrator/evals
orchestrator/src/orchestrator/artifacts.py` empty; `def _unwrap_verdict_envelope` gone.
Deps: γ (same files), δ (`reviewer_trial/__main__.py`).

**η — Reviewer-trial drivers: one panel cache, one run-and-score loop** · low · ~600 LOC ·
keys `fk-294ee8dfa70f`
Files: `evals/reviewer_trial/runner.py` (`PanelRunResult.from_dict`, one cache function,
`run_and_score`), `evals/reviewer_trial/__main__.py` (six score loops and `_load_or_run_panel`
removed), `evals/reviewer_trial/mining.py`, `evals/reviewer_trial/adjudication.py` (spot-check
stickiness and re-split from `mine`), tests (`test_reviewer_trial_main.py`,
`test_reviewer_trial_runner.py`, `test_reviewer_trial_mining.py`).
Signal: `git grep -n "_load_or_run_panel\|Mirrors the caching logic"
orchestrator/src/orchestrator/evals` empty; a smoke run followed by a full run over the same
fixture re-uses the cache only when prices, gate and diff cap match (the divergence the
finding names is gone, asserted by test); `mine` is a thin CLI over `mining`/`adjudication`.
Deps: ζ, γ.

**θ — `report.py` leaves: stats, prices, cell admission, table rendering, Elo** · low ·
~1,200 LOC (move-dominated) · keys `fk-b2c28b4f8f17` (part)
Files: `evals/stats.py`, `evals/prices.py`, `evals/cell_admission.py`, `evals/table_render.py`,
`evals/elo_report.py` (all new), `evals/report.py`, `evals/judge_pilot.py`, importers in
`orchestrator/cli.py`, tests (`test_eval_composite_report.py`, `test_eval_driver_report.py`).
Signal: each new module's docstring states one purpose and it imports nothing from `report`;
`report.py` no longer holds Elo, t-statistics or price tables.
Deps: ε, β.

**ι — `report.py` surfaces: composite and plan-quality; rationale to docs** · low · ~1,500
LOC (move-dominated) · keys `fk-b2c28b4f8f17`
Files: `evals/composite_report.py`, `evals/plan_quality_report.py` (new), `evals/report.py`
(deleted), `docs/plan-scoring-and-judge.md`, importers (`orchestrator/cli.py`,
`evals/judge_pilot.py`, `scripts/run_fable_trial_v2_campaign.py`,
`scripts/run_judge_ofat_pilot.py`), tests (`test_eval_composite_report.py`,
`test_eval_architect.py`, `test_judge_ofat_pilot.py`).
Signal: `orchestrator/src/orchestrator/evals/report.py` does not exist; the report snapshot
over the packaged dir is unchanged (diffed in the leaf's verification note), including the
three legacy cap-exhausted cells in the `cap_exhausted` bucket.
Deps: θ.

**κ — Eval CLI as a click group in `evals/cli.py`** · low · ~1,300 LOC (move-dominated) ·
keys `fk-4352b1ee49f5`
Files: `evals/cli.py` (new), `orchestrator/cli.py`, `orchestrator/cli_config.py` (new,
config-preamble decorator), `evals/runner.py` (drop `from orchestrator.cli import eval_cmd`),
tests (`test_eval_configs.py`, `test_eval_driver_cli.py`, `test_eval_replay_frame.py`,
`test_eval_architect.py` patch targets).
Signal: `orchestrator eval --help` and every eval subcommand's `--help` list the same options
as before; `git grep -n "ConfigRequiredError" orchestrator/src/orchestrator/cli.py` shows one
handler; `_run_plan_only` emits a `routing_decision` for role `architect` (test); `git grep -n
"orchestrator.cli import" orchestrator/src/orchestrator/evals` empty.
Deps: ι, δ, ε.

**λ — One checkout root for eval defaults** · low · ~300 LOC · keys `fk-14926fe151d4`
Files: `evals/paths.py` (new), `evals/reviewer_trial/runner.py`,
`evals/reviewer_trial/scorer.py`, `evals/reviewer_trial/mining.py`,
`evals/reviewer_trial/corpus.py`, `evals/reviewer_trial/__main__.py`,
`evals/prompt_opt/optimizer.py`, `evals/prompt_opt/__main__.py`,
`evals/prompt_opt/curator_corpus.py`, tests (`test_eval_paths.py` new).
Signal: `git grep -n "/home/leo" -- 'orchestrator/src/orchestrator/evals/*.py'` matches only
comments/docstrings; with S14's root override pointed at a tmp checkout,
`reviewer_trial`'s default runs-db and repo-root resolve under it (test).
Deps: η, 6597, S14's `shared/project_layout.py` leaf.

DAG: α → β → γ → δ → ε → θ → ι → κ; γ, δ → ζ → η → λ; β → θ; δ, ε → κ. γ → δ is a
same-file edge (`reviewer_trial/__main__.py`), not a semantic one. Highs: γ, δ. External: λ →
6597 and S14's project_layout leaf; α, ε, ι → S25's bootstrap leaf where it declares the same
scripts; κ → any other stream's leaf declaring `orchestrator/cli.py`.

### G7 walk (design invariants)

- INV-11 `no-silent-fail-soft` — the stream's purpose; γ's reducers carry the count in the
  result, not a log. Pass.
- INV-13 `readers-prove-their-producer` — reducers' unmeasured counts ship in the same leaf
  as their producer (`measured_call`); persisted-cell readers have a stated backfill decision
  per record kind (D3). Pass.
- INV-5 / INV-9 — every leaf removes a lock-step copy; no leaf adds one. ζ deliberately leaves
  `workflow._run_reviewer`'s unwrap (recorded consumer, §9). Pass.
- INV-2 `structured-facts-at-failure` — β persists the cause the classifier held instead of a
  string the report re-parses. Pass.
- INV-12 `exceptions-owned-or-ratified` — β removes a `noqa: PLC0415`; no leaf adds an
  inline suppression. Pass.
- INV-1, 3, 4, 6, 7, 8, 10 — not engaged.

### META

Decomposed and queued without further oversight, this batch yields one measurement vocabulary
(C1) that every scoring site and reducer shares, one owner for the persisted cell (C2), one
review reading shared with production (C3), and an architect cell whose policies are separately
testable (C4), each landed with a two-way boundary test, in an order that serialises every
shared file. No design question is left open; §12 is tactical.

## 12. Open questions (tactical)

1. γ: whether the pairwise-judge and assessor callers can obtain a usage gate from their CLI
   drivers (today none is built); `measured_call` takes one when offered, and wiring it is
   optional in γ.
2. λ: if S14's `project_layout` leaf lands without a reconciliation `tickets_db` locator,
   λ files the request on S14's leaf and keeps `CURATOR_TICKETS_DB` until it lands.
3. κ: the `cli_config` decorator touches non-eval commands in `cli.py`; decompose checks other
   streams' filed leaves on `orchestrator/cli.py` and wires same-file edges.
