# Capability manifest — adaptive-model-routing-prd

Bindings verified on main @ `8a576c608a` (2026-07-13). Mechanizes G3+G6 per leaf:
every capability a task's signal asserts is bound to evidence — a grep on main or an
upstream producer task in the dependency closure. FAIL values would block the batch;
**all bindings below PASS** (one deliberate probe-gated admission noted at ξ).

Shared substrate (consumed by several tasks):

| Capability | Evidence |
|---|---|
| single model-resolution chokepoint w/ role-key split | grep:orchestrator/src/orchestrator/workflow.py:7351 `role_key = role.name.split('_')[0]` — wired in `_invoke`, the production invocation path |
| existing dynamic-selector seam | grep:workflow.py:7282 `def _select_model_for_role` (Rust heuristic reads plan shape at :7294-7298 `step_count = len(self.plan.get('steps', []))`) |
| dead simple_task config fields | grep:orchestrator/src/orchestrator/config.py:1903-1904 `simple_task_budget_usd` / `simple_task_max_turns` (defined; no consumer — declared-only is the DEFECT α fixes, not an assumed capability) |
| hot-reload auto-registration for new submodel fields | grep:config.py:2779 `_submodel_leaf_paths` + :2801-2803 RELOADABLE_FIELDS entries |
| model string passes verbatim to CLI | grep:shared/src/shared/cli_invoke.py `--model` argv assembly (survey-verified :1225-1251) — arbitrary IDs (`claude-fable-5`) transit today |
| per-invocation model+role+cost telemetry | producer:task-295 (done) — `invocations` table schema (`model TEXT NOT NULL`, `role TEXT NOT NULL`) in shared/src/shared/cost_store.py; rows live in data/orchestrator/runs.db (12,295 rows, survey) |
| event vocabulary + emit path | grep:orchestrator/src/orchestrator/event_store.py:48-51 `invocation_end` / `phase_enter` — new `routing_decision` type is additive |
| typed-metadata extension point | grep:shared/src/shared/task_metadata.py:50,69 `ConfigDict(extra='allow')` + typed-field precedent (Milestone/DoneProvenance) |
| daily-ceiling machinery precedent | grep:orchestrator/src/orchestrator/defaults.yaml:470-471 `watcher_daily_cost_ceiling_usd` / `orch_daily_cost_ceiling_usd` |
| workflow test rig injectable runner | grep:workflow.py `invoke_fn` seam (:7439-7452 region; same seam mcp-verdict θ=2488 drives) |

Per-leaf bindings (leaves: θ, ι, κ, ξ, ρ, σ; intermediates listed where their
signal asserts capability beyond the shared table):

## α (intermediate — role-key retirement)
- config submodels enumerate roles w/o simple_task → grep:config.py ModelsConfig:140-152 (absence confirmed by survey; α adds the fields) — defect-fix, no absent-substrate assumption.
- consumers: ε (resolver), λ (turns tuning) — in-batch, downstream. PASS.

## β (intermediate — allowlist/validation/probe)
- fail-fast classification home → grep:cli_invoke.py API-error classification (survey :467-470 "transient — worth retrying"); β adds a model-not-found subclass. PASS (site exists).
- probe substrate: account pool config/usage-accounts.yaml (6 accounts) + `claude --model <id>` verbatim transit (shared table). PASS.
- reload validation surface → RELOADABLE_FIELDS + reload_config structured dispositions (config-hot-reload PRD, shipped). PASS.

## γ (intermediate — routing_decision persistence)
- event emit + metadata write paths: shared table rows 7 & 8. `metadata.routing` is a NEW typed field (producer: γ itself); consumers μ/ν/δ are downstream in-batch. PASS.

## δ (intermediate — per-(model×role) rollup)
- steward/triage invocation rows → **producer:task-2461 upstream (dep wired at filing)**; judge rows already present (2461's correction — grep:workflow.py:7538 chokepoint records `role=role.name`). PASS.
- module_tagger/unblock_auto cost threading → verify-at-impl; grep:harness.py:1921-1923 (module_tagger passes model/turns/budget but cost_store threading unverified) — δ's scope includes threading IF missing; not an assumed capability. PASS.
- digest + dashboard homes → orchestrator/digest.py, dashboard/ (existing surfaces). PASS.

## ε (intermediate — resolve_route + policy table)
- chokepoint + selector seam + hot-reload + ceiling precedent: shared table rows 1, 2, 4, 9. PASS.
- byte-equivalence fixture basis: current values enumerable from defaults.yaml:158-220 + roles.py dataclass defaults (survey table). PASS.

## ζ (intermediate — typed model_overrides)
- shape-validation precedent at submit → grep:fused-memory/src/fused_memory/server/tools.py:3071 `deterministic_task_error` guard pattern. PASS.
- resolve-time string validation fail-safe → ε layer-skip (in-batch upstream). PASS.

## η (intermediate — out-of-band adoption)
- steward → grep:steward.py:544 `model=self.config.models.steward`, :575 `backend=self.config.backends.steward`; deep_reviewer → grep:review_checkpoint.py:183-185 `getattr(self.config.models, 'deep_reviewer', 'opus')`; module_tagger → grep:harness.py:1921; unblock_auto → grep:dry_run_unblock.py:273 `ua_cfg = config.unblock_auto`, :314 `role='unblock_auto'`. All four sites exist + wired. PASS.
- Lock-note: mcp-verdict ε=2485 co-edits steward.py:599-668 (different concern; module locks serialize). PASS.

## θ (leaf — plan-shape fleet rules)
- plan shape available at selection: grep:workflow.py:7294-7295 (`self.plan.get('steps')` in `_select_model_for_role`) → generalization has live substrate. Signal (routing_decision names the rule on a non-Rust ≥12-step plan) producible by ε (upstream) + config rule (this task). PASS.

## ι (leaf — architect effort/model adoption)
- measurement instrument → **producer:task-2475 (architect eval coverage) + producer:task-2478 (OFAT→matrix→confirm driver), both upstream via wired deps** (eval-revival batch, pending). Config-flip machinery: hot-reload green tier (shared row 4). Decision record: committed artifact — no numeric premise asserted (pass/marginal band decided from the eval report; G6 branch 1 N/A by construction). PASS (producer-upstream).

## κ (leaf — haiku pilots judge/triage/module_tagger)
- valid full-workflow instrument → **producer:task-2472 (eval Phase-1 gate: today the framework grades EMPTY DIFFS — D1) upstream**. PASS.
- post-migration contracts → **producer:task-2486 (judge verdict tool) + producer:task-2485 (triage verdict tool) upstream**. PASS.
- replay substrate for triage/module_tagger: historical inputs in data/escalations/ (~2,352 files) + runs.db events; module_tagger call site grep:harness.py:1921. PASS.
- thresholds derive from δ's measured baselines (upstream dep) — no guessed numeric bound (G6 branch 1). PASS.

## λ (intermediate — simple_task tuning)
- tunable fields → producer:α (upstream). Saturation metric → producer:δ (upstream). 28.6% baseline: measured (survey invocation_end analysis), not asserted as a target. PASS.

## μ (intermediate — retry-escalation rung)
- dispatch bookkeeping + redo path → grep:harness.py:5935 `_maybe_auto_eval` (extends); escalation flag → grep:escalation/src/escalation/server.py:132 `RESOLVE_ACTIONS` tuple (additive flag). Tier state → producer:γ (upstream). Ladder arithmetic → producer:ε. PASS.

## ν (intermediate — saturation stamp)
- simple-task outcome handling → grep:workflow.py:3311 `_run_simple_task`, :3420/:3446 `_stamp_optimistic_path` (stamp-write precedent). Turn-cap detection: invocation_end turns vs max_turns (event exists, shared row 7). Policy rule → producer:ε. PASS.

## ξ (leaf — fable admission)
- **Deliberately probe-gated**: fable availability on pool accounts is NOT bound to on-main evidence — it is bound to **producer:β's probe artifact (upstream hard dep)**, per G3 resolution (b). Admission step refuses on probe FAIL. Ceiling knob → producer:ε; override path → producer:ζ; retry rung → producer:μ. PASS (explicit-prerequisite form).

## ρ (leaf — docs)
- Doc homes exist: CLAUDE.md, skills/orchestrate/SKILL.md. PASS.

## σ (leaf — integration gate)
- Rig + fake-runner seam: shared table row 10 (same seam 2488 uses). All exercised
  mechanisms produced by upstream in-batch intermediates (ε, ζ, η, μ, ν, δ). PASS.

## Rejection-mechanism bindings (G6 branch 4)
- "submit with malformed model_overrides is rejected" (ζ signal): rejection mechanism = the submit_task guard ζ itself adds, tested by authoring the malformed submit and observing the ValidationError — same shape as the existing deterministic_task_error guard (tools.py:3071), which observably fires today. PASS (mechanism-with-precedent, bound by ζ's own RED test).
- "reload with unknown rule key returns ValidationError" (ε signal): precedent = shared Milestone model validation rejecting malformed specs at submit (task_metadata.py) + reload_config structured error dispositions (shipped hot-reload PRD). PASS.
- "invalid override model falls through + recorded" (ζ/ε): fail-safe skip is ε's invariant 2, exercised in σ test 2. PASS.
