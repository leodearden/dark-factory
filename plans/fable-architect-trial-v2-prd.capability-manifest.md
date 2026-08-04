# Capability manifest — fable-architect-trial-v2-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized), verified against
main 2026-08-04. Machine-readable twin:
`plans/fable-architect-trial-v2-prd.capability-manifest.yaml`. The four
instrument tasks (ρ, σ, υ, φ) are **eval-framework-revival lane** tasks
(its decision 11; paired edit 2026-08-04) — their stampable sidecar entries
live in `plans/eval-framework-revival-prd.capability-manifest.yaml`; they are
summarized here so this batch's substrate story is complete in one artifact.

## §ρ — architect-fable-max candidate config (revival lane)

- `ARCHITECT_EVAL_CONFIGS` list + `get_config_by_name` resolver searches it —
  wired: `orchestrator/src/orchestrator/evals/configs.py` (list at ~:435;
  resolver iterates `*ARCHITECT_EVAL_CONFIGS`). **PASS**
- Effort-sensitivity premise (+0.0317 opus high→max inside the v1 screen) —
  reproduced from raw dumps 2026-08-04; π's own text anticipated this
  variant "if the consumer campaign's screen shows effort sensitivity".
  **PASS** (numeric premise empirically anchored)

## §σ — reference back-fill + judged_without_reference marker (revival lane)

- `post_task_commit` present in all three fixtures (`reify_task_12`,
  `reify_task_27`, `df_task_18`) with no `reference` block — verified by
  loading each JSON 2026-08-04. **PASS** (back-fill source exists)
- `get_diff_between_commits` — wired: called by `run_architect_eval`
  (`runner.py` step 6). **PASS**
- Marker consumer: per-config report accumulator (`report.py` `by_config`)
  is the established aggregation point (cap_excluded precedent, tasks
  3118/3302/3379). **PASS**

## §υ — judge cost recorded on architect cells (revival lane)

- `metrics.judge_cost_usd` field exists (`metrics.py:72`, default 0.0) and
  the report layer already aggregates + surfaces it (`report.py:722,808`) —
  the gap is solely `run_architect_eval` never populating it (grep:
  `judge_cost_usd` absent from `runner.py`). **PASS** (field-population task;
  producer path exists, consumer wired)
- Judge invocation returns cost: agent invocations return `AgentResult` with
  `cost_usd` (the architect call reads `result.cost_usd` at the same call
  shape). **PASS**

## §φ — UsageGate on run_architect_eval (revival lane)

- `invoke_with_cap_retry(usage_gate, label, **invoke_kwargs)` exists
  (`shared/src/shared/cli_invoke.py:1009`) with bounded patience
  (`cap_wait_sanity_secs` / `max_cap_retries` → `AllAccountsCappedException`)
  and a standalone non-workflow caller precedent
  (`orchestrator/src/orchestrator/dry_run_unblock.py:340`). **PASS**
- Gate construction pattern: `run_eval` (`runner.py:397-402`) —
  enabled-guarded, warn-and-degrade. **PASS**
- Scope substrate: `scoped_cap_models` defaults `['claude-fable-5']`
  (`shared/src/shared/config_models.py:74`); scope-aware slot loop live in
  `cli_invoke.py`. **PASS** (usage-gate-model-scoped-caps PRD, done)
- Must-not-touch pinned (G6 negative twin): the predicate
  `tainted = arch_unmeasurable and not is_scorable_plan(plan)`
  (`runner.py:731`) and timeout-not-tainted semantics stay byte-identical;
  the sidecar pins the predicate's survival with an `expect: present` grep.

## §β1 — curated v2 hard fixture pool

- Minting substrate: `build_fixture_record` (`task_sampler.py:421`),
  `capture_reference` (`:512`), `pin_eval_branch` (`:557`) all exist. **PASS**
- Census substrate: `json_extract(events.data,'$.subtype')` on
  `invocation_end` rows across per-project `runs.db` — census run
  2026-08-04 (41 distinct tasks at 121-turn exhaustion; no retention
  trimming, `sqlite_sequence.seq == COUNT(*)` all 7 DBs). **PASS**
- Merge-SHA availability: sampled 32 reify candidates all present, nearly
  all `done` ⇒ `Merge task/<id>` commit exists for `capture_reference`;
  SPLIT minority minted planRate-only. **PASS** (with the planRate-only
  degradation explicit, not assumed away)
- Default-run isolation: `cli._load_fixture_dir` globs `*.json`
  **non-recursively**, so `tasks_hard_v2/` can never leak into default eval
  runs. **PASS** (structural)

## §β2 — committed v2 campaign driver

- Copy source exists: `data/eval-campaign/fable_architect_only.py` (v1
  driver, main checkout). **PASS**
- Per-candidate cap_excluded substrate already in the report layer
  (`plan_quality_cap_excluded` in the per-config accumulator) — driver
  surfaces, does not build. **PASS**
- `judged_without_reference` counts: producer is σ (upstream). **PASS**
  (producer:σ, DAG-direction correct)

## §γ1 — stage-1 calibration run gate

- Producers all upstream: σ, υ, φ (instrument), β1 (fixtures), β2 (driver).
  **PASS** (DAG-direction)
- planRate premise: `plan_steps` persisted per cell (`runner.py:846`,
  `len(plan.get('steps') or [])`) — judge-free, valid without reference.
  **PASS**
- Q_ceiling derivability: v1 raw dumps exist (`data/eval-campaign/`,
  reproduction recipe decision-record §8, reproduced digit-for-digit
  2026-08-04); incumbent cells on validly-referenced fixtures are the
  anchor population. **PASS** (provisional threshold, γ2-ratified — G6
  resolution (b))
- Cell economics: incumbent $3.28/cell observed + judge cost now recorded.
  **PASS**

## §γ2 — banding ratification + regime ruling + stage-2 authorization gate

- Pure owner gate (2864 shape: `task_kind='deterministic'`,
  `always_escalates`); consumes γ1's committed report. The equal-cost vs
  equal-turns ruling is deliberately un-defaulted (Leo raised it, did not
  rule); all three regime options carry fully-specified recipes in the PRD
  (D5), so the resolution is executable whichever way it rules. **PASS**

## §δ — stage-2 screen run gate

- `architect-fable-max` producer: ρ (upstream). **PASS**
- Harness bypasses `routing.allowed_models`: `run_architect_eval` calls
  `invoke_agent` directly with `model=config.model`; the allowlist check
  lives only in `routing_dispatch` — no admission needed to run fable.
  **PASS**
- Budget knob per regime: `EvalConfig.max_budget_usd` (default 20.0) is
  per-config, `max_architect_turns` per-fixture — both rulings are pure
  parameter choices, no runner surgery. **PASS**

## §ζ — v2 decision record

- Analysis inputs: δ's results artifact; validity bounding via σ's marker;
  cap symmetry via per-config cap_excluded. All upstream. **PASS**
- v1-vs-v2 cost comparability caveat (v1 excluded judge cost) recorded as a
  mandatory statement in the record. **PASS**

## §η — admission re-ruling gate

- Pure owner gate naming ζ's record (the τ3/2864 shape). On ADMIT the
  resolution instructs filing a NEW flip task — 2544 is cancelled and is
  never resurrected (D12). **PASS**
