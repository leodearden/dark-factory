# PRD: Adaptive Model Routing (Survey Tiers 1–3)

**Status:** active — authored 2026-07-12.
**Predecessor:** `plans/author-declared-complexity-prd.md` (shipped), `plans/task-routing-survey-2026-07-12.md` (evidence base).

## Goal

Route tasks and individual workflow steps to a wider range of models — `haiku` → `sonnet`
→ `opus` → `claude-fable-5` — chosen per task and per phase instead of by a static
per-role table. An operator can: (a) pin a model per role per task via typed metadata;
(b) install fleet-wide routing rules via hot-reloadable config; (c) see every routing
decision, and per-(model×role) outcome rates, in the digest and dashboard. The pipeline
itself escalates to a stronger model on re-dispatch after failure and stops re-attempting
the simple path once it has demonstrably saturated.

## Background

The 2026-07-12 routing survey found: routing is static and submit-time-only; the sole
dynamic hook is a hardcoded Rust-only sonnet→opus implementer upgrade
(`workflow.py:7282`); rich difficulty signals (turn-cap saturation, verify attempts,
retry ledgers) are persisted but never read back; a failed task always retries with the
identical path+model. Evidence highlights: simple_task saturates its 30-turn cap 28.6%
of runs yet is un-tunable (config-key derivation bug); architect consumes 47% of
all-time spend at a 0.5% cap rate while the sonnet implementer caps at 9.2%; the
reviewer_trial eval harness already drove one successful routing change (5×sonnet panel
→ 1×opus reviewer, 6.3× F1/$).

All per-task LLM calls funnel through `TaskWorkflow._invoke`
(`orchestrator/src/orchestrator/workflow.py:7351-7359`); per-role model config is
hot-reloadable green tier; per-invocation model+cost telemetry exists in
`data/orchestrator/runs.db` `invocations`.

## Sketch of approach

One new pure function, the **route resolver**, becomes the single authority for
(model, effort, budget_usd, max_turns) at every LLM invocation:

```
resolve_route(role, task_metadata, plan_shape, routing_state, config) -> RoutingDecision
```

Layered precedence (first hit wins):
1. **`task.metadata.model_overrides[role]`** — typed, explicit, validated per-task pin.
2. **Policy table** (`routing.rules` in orchestrator config, hot-reloadable) — ordered
   rules matching on a closed condition vocabulary; first match supplies overrides.
3. **Static per-role table** (`config.models.<role>` etc.) — today's behaviour.
4. **Role dataclass defaults** — today's fallback.

Adopted by `_invoke` and the four out-of-band dispatch sites (steward + its triage,
deep_reviewer, module_tagger, unblock_auto). Every resolution emits a persisted
**`routing_decision`** record (which layer/rule fired, resolved values, tier counter).

Substrate shipped first (Phase 1, from survey Tier 0): simple_task made
config-addressable; model allowlist + validation + fail-fast + per-account availability
probe; routing-decision persistence; per-(model×role) outcome rollup in digest+dashboard.

Adaptive layer (Phase 4, survey Tier 3): a per-task **routing tier counter** bumped on
re-dispatch-after-failure and auto-eval redo drives a `retry → next tier up` policy rule
(ladder capped at `claude-fable-5`); simple_task turn-cap saturation stamps the task so
subsequent dispatches take the full path.

Fable is admitted top-of-ladder from day one, gated on the availability probe and a
per-model daily USD ceiling; initial policy reaches it only via explicit override or the
retry ladder's final rung.

## Resolved design decisions

1. **Tier-0 substrate folded into this PRD** (not a separate PRD): Phase 1 tasks are
   hard prerequisites in the DAG.
2. **Layered routing authority**: metadata override > policy table > static config >
   role default. Both explicit control and fleet-tunable automation; Tier-3 hooks are
   just policy rules over persisted routing state.
3. **Model menu**: mechanism accepts arbitrary validated model strings (CLI aliases and
   pinned IDs). Initial allowlist: `haiku`, `sonnet`, `opus`, `claude-fable-5`. Ladder
   order for tier arithmetic: haiku < sonnet < opus < claude-fable-5.
4. **Fable from day one, conservatively**: admission gated on the per-account
   availability probe (this is unverified substrate — only the interactive account is
   proven); reachable initially only via `model_overrides` or the retry ladder's final
   rung; bounded by a per-model daily USD ceiling (extends existing daily-ceiling
   machinery, `defaults.yaml` cost-ceiling block). Per-(account,model) cap-state rework
   in the UsageGate is explicitly out of scope — a fable cap-hit CAPs the whole account
   (existing semantics), which the ceiling + failover make survivable.
5. **Plan-shape-derived routing now; architect hint later**: policy conditions read what
   the plan already exposes (`step_count`, module footprint — generalizing the existing
   Rust heuristic to any language). An explicit architect difficulty-hint plan-tool
   field is a named follow-up (see Open questions), to be designed once the rollup
   shows where shape-derived routing mis-fires.
6. **Retry-escalation trigger semantics**: the routing tier counter increments on
   (a) re-dispatch after a prior terminal-failed dispatch of the same task (outcome
   blocked, or requeued after losing completed work), and (b) auto-eval redo dispatch.
   It does NOT increment on within-workflow verify/review iterations or steward L0
   resumes (same dispatch) — those would burn top-tier quota on routine debugging.
   `resolve_issue` gains an optional `escalate_model` flag that pre-increments the tier
   for the next dispatch.
7. **Resolver adoption breadth**: `_invoke` plus ALL out-of-band sites (steward + inner
   triage, deep_reviewer, module_tagger, unblock_auto) resolve through `resolve_route`.
   Divergence of the out-of-band sites is a documented failure mode (steward.py:556-569).
8. **Byte-equivalent default**: the resolver ships with a default policy that reproduces
   current behaviour exactly (including the Rust heuristic as rule #1) so Phase 2 lands
   with zero fleet behaviour change; fleet rule changes happen in Phase 3 with the
   rollup watching.
9. **Validation split**: `submit_task` validates `model_overrides` **shape** (known role
   names, string values); the orchestrator validates model **strings** against its
   allowlist at resolve time (fused-memory does not know the orchestrator's allowlist).
   Unknown model at resolve time → fall back one layer + WARN + `routing_decision`
   records the rejection (fail-safe, never fail the dispatch).
10. **Decide-and-act pilots**: Tier-1 config changes (haiku on mechanical roles,
    architect effort trial, simple_task turns) are single tasks that run the trial,
    apply the flip on a clear pass, and escalate only on a marginal band — never
    standalone no-code decision tasks (orchestrator-churn lesson).
11. **Measurement/adoption split with eval-framework-revival (reconciled
    2026-07-13)**: eval-revival OWNS the measurement instrument (profile 2466,
    architect coverage 2475, OFAT→matrix→confirm driver 2478) and explicitly
    leaves production config adoption out of its scope; THIS PRD owns adoption
    (tasks ι, κ). This PRD makes NO edits to `evals/runner.py`, the benchmark
    suite, or the judge/Elo machinery. The earlier draft's "un-hardcode
    build_eval_orch_config" is withdrawn — that is eval-revival β=2466.
12. **Resolution precedes prompt assembly** (tier1-prompt-optimization seam,
    its P-4): `resolve_route` runs before the role system prompt is built in
    `_invoke`, and the resolved model is available at the prompt-build point —
    the tier1 prompt-artifact loader keys heuristics artifacts on the
    router-resolved executor model. Owned here (invariant 9); tier1's loader
    call site consumes it.

## Pre-conditions for activating

- Phase 1–2 have no external pre-conditions beyond this PRD's own Phase-1 tasks.
- Cross-PRD task deps (all intra-`dark_factory`, bare-integer, wired at
  decompose): δ → **2461** (harness-backend T6: steward/triage CostStore rows —
  the judge already records, per 2461's correction); ι → **2475** + **2478**
  (eval-revival architect coverage + OFAT/confirm driver); κ → **2472**
  (eval-revival Phase-1 validity gate — the framework currently grades empty
  diffs), **2486** + **2485** (mcp-verdict judge/triage contract migrations —
  trial on the post-migration transport, not the dying one).
- Assumed-substrate verified during authoring: `_invoke` chokepoint
  (workflow.py:7351-7359), plan shape available at selection time (workflow.py:7295),
  `retry_ledger`/typed-metadata pattern (shared/src/shared/task_metadata.py), event
  store (orchestrator/src/orchestrator/event_store.py), invocations telemetry with
  per-invocation `model` (shared/cost_store.py), daily cost-ceiling machinery
  (defaults.yaml cost-ceiling block), green-tier hot-reload via `_submodel_leaf_paths`
  (config.py:2779-2807), eval harness (evals/runner.py, reviewer_trial/). Fable
  availability on pool accounts is the one UNVERIFIED capability → probe task β is a
  hard prerequisite of fable admission (task ξ).

## Cross-PRD relationship

Reconciled 2026-07-13 against the four sibling PRDs committed 2026-07-12
(all decomposed with live task ids).

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/config-hot-reload-prd.md` (shipped) | extends | new `routing.*` config block joins green-tier `RELOADABLE_FIELDS` | this-prd | queued (task ε) |
| `plans/author-declared-complexity-prd.md` (shipped) | extends | simple-path fallback story gains saturation stamp; declared-complexity semantics unchanged | this-prd | queued (task ν) |
| `plans/harness-backend-reconnect-pi-prd.md` (T1=2457 **done**, T5=2460, T6=2461, pi=2463) | consumes 2461; boundary on the rest | δ consumes 2461's steward/triage CostStore rows (judge already records — do NOT re-instrument). Backend axis stays theirs: the resolver returns model/effort/budget/turns, NEVER backend; `backends.<role>` selection + provider pools + `_MODEL_COSTS`→prices are that PRD's. 2460 co-edits `_build_agent_env` (workflow.py:7304-7333) adjacent to ε — lock-serialized, no semantic overlap | each PRD its axis; **2461** owns steward/triage cost threading | δ dep wired at decompose |
| `plans/eval-framework-revival-prd.md` (β=2466, ι=2472, ζ=2473, θ=2475, λ=2477, μ=2478, ν=2479, ξ=2480) | consumes | measurement/adoption split (decision 11): they measure, this PRD adopts. ι deps 2475+2478; κ deps 2472. This PRD never edits `evals/runner.py`/benchmark/judge machinery. Endpoint-swapped model bundles (per-role ANTHROPIC_BASE_URL) are their ν/ξ — a separate widening axis composing with this PRD's model axis | **eval-revival** owns the instrument; **this-prd** owns adoption flips | ι/κ deps wired at decompose |
| `plans/mcp-verdict-servers-prd.md` (α=2481 in-progress, β=2482, ε=2485, ζ=2486) | consumes (contract stability) | κ trials judge/triage on the POST-migration verdict-tool transport → κ deps 2486 (judge) + 2485 (triage). 2482 injects verdict-tools at the `_invoke` spawn site; η co-edits steward.py near 2485 — lock-serialized | **mcp-verdict** owns the transport | κ deps wired at decompose |
| `plans/tier1-prompt-optimization-prd.md` (loader keys artifacts on executor model; its P-4 names this PRD) | produces | invariant 9 / decision 12: `resolve_route` completes before prompt assembly; the resolved model is available to the prompt-artifact loader call site. Per-model artifact sets make a routed reviewer/curator load the right heuristics block by construction | **this-prd** owns exposing the resolved model; **tier1** owns the loader + call site | no hard dep either way (their analysis, concurred) |
| `plans/dashboard-taskgraph-legibility-prd.md` | none | disjoint dashboard surfaces (`tab_tasks.jsx` vs δ's new cost/routing panel) | — | no seam |
| `plans/afk-digest.md` | produces | rollup + routing sections consumed by digest | this-prd | queued (task δ) |

No contested ownership; the one bidirectional-looking seam (tier1 loader ↔
resolver) is split by mechanism: resolution-ordering owned here, loader owned
there.

## Contract — route resolution seam (approach B+H)

### Types

```python
# orchestrator/src/orchestrator/routing.py  (new module)

@dataclass(frozen=True)
class RouteInputs:
    role_name: str                    # full role name, NOT split-derived key
    task_id: str
    task_metadata: Mapping[str, Any]  # includes model_overrides, complexity, routing state
    plan_shape: PlanShape | None      # step_count, module_paths; None pre-plan
    routing_tier: int                 # persisted per-task counter, default 0
    dispatch_count: int

@dataclass(frozen=True)
class RoutingDecision:
    model: str
    effort: str
    budget_usd: float
    max_turns: int
    source_layer: Literal["metadata_override", "policy_rule", "config", "role_default"]
    rule_id: str | None               # policy rule name when source_layer == policy_rule
    rejected: tuple[str, ...]         # e.g. ("metadata_override:model-not-in-allowlist",)

def resolve_route(inputs: RouteInputs, config: OrchestratorConfig) -> RoutingDecision: ...
```

### Config schema (`routing:` block, all green-tier hot-reloadable)

```yaml
routing:
  allowed_models: [haiku, sonnet, opus, claude-fable-5]
  ladder: [haiku, sonnet, opus, claude-fable-5]     # tier arithmetic order
  per_model_daily_ceiling_usd: {claude-fable-5: 50.0}
  rules:                                            # ordered; first match wins
    - id: rust-large-plan-implementer               # byte-equivalent default rule
      match: {role: [implementer, debugger], plan_min_steps: 12,
              plan_min_modules: 3, module_prefix: "crates/"}
      set: {model: opus}
    # Phase-3 fleet rules land here via hot-reload, e.g.:
    # - id: retry-tier-up
    #   match: {role: [implementer, debugger, architect], min_routing_tier: 1}
    #   set: {model: "+1"}                          # ladder-relative bump, capped at top
```

Condition vocabulary is CLOSED: `role`, `task_complexity`, `task_priority`,
`plan_min_steps`, `plan_min_modules`, `module_prefix`, `min_routing_tier`,
`min_dispatch_count`, `simple_saturated`. Unknown keys → config ValidationError at
load/reload (never silently ignored). `set` may carry any of
`model` (absolute or ladder-relative `"+N"`), `effort`, `budget_usd`, `max_turns`.

### Invariants

1. **Total**: `resolve_route` never raises for well-formed config; for every input it
   returns a decision (worst case: role_default layer).
2. **Fail-safe validation**: a model string not in `allowed_models` at any layer causes
   that layer to be skipped, the rejection recorded in `rejected`, and resolution to
   continue down the layers. A dispatch is never blocked by a routing mis-config.
3. **Byte-equivalence default**: with an empty `model_overrides`, the shipped default
   `rules`, and unchanged `models.*` config, `resolve_route` returns exactly what
   today's `_invoke` + `_select_model_for_role` produce for every role — including
   `simple_task` (whose current values become config-visible rather than dataclass-only).
4. **Full role names**: resolution keys on `role_name` verbatim; the `split('_')[0]`
   derivation is retired. `models.simple_task`, `budgets.simple_task` etc. become real
   config fields; existing `simple_task_budget_usd`/`simple_task_max_turns` are honored
   or formally deprecated in the same change (not left dead).
5. **Ladder arithmetic**: `"+N"` bumps clamp at the ladder top; a model absent from
   `ladder` cannot be bumped from (treated as its allowlist position if present, else
   rule skipped + recorded).
6. **Ceiling**: when `per_model_daily_ceiling_usd[m]` is exhausted (cost_store day
   window), any decision resolving to `m` falls back one layer (same mechanics as
   invariant 2) — ceilings bound spend, they don't block dispatch.
7. **Every invocation emits exactly one `routing_decision` event** (event_store) carrying
   the full `RoutingDecision` + inputs digest, and the latest decision is mirrored to
   `task.metadata.routing` (bounded history, newest N=5).
8. **Tier monotonicity**: `routing_tier` only increments (per decision 6 triggers) and
   never resets within a task's lifetime; stamped via the same metadata write path the
   harness already owns (never author-supplied).
9. **Resolution-before-prompt**: within `_invoke`, `resolve_route` completes before the
   role system prompt is assembled, and the `RoutingDecision` (specifically `.model`) is
   in scope at the prompt-build point — the tier1 prompt-artifact loader keys heuristics
   artifacts on the resolved executor model and must never compose against a stale
   static-config model.

### Boundary-test sketch

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Metadata override wins | task metadata `model_overrides: {implementer: haiku}`; policy + config would say sonnet | implementer invocation runs haiku; `routing_decision.source_layer == metadata_override` |
| 2 | Invalid override falls through | `model_overrides: {implementer: gpt-9}` (not in allowlist) | invocation runs the policy/config model; `rejected` names the override; WARN logged; dispatch proceeds |
| 3 | Byte-equivalence | no overrides, shipped default rules, stock `models.*` | for each of the 12 roles, resolved (model, effort, budget, turns) equals the pre-PRD values captured as a fixture; Rust-heuristic rule fires identically on a ≥3-crates/≥12-step plan |
| 4 | simple_task config-visible | `models.simple_task: haiku` set via `reload_config` | next simple_task invocation row shows haiku; reload `applied` disposition lists the field |
| 5 | Retry tier bump | task's dispatch #1 ends blocked; `retry-tier-up` rule installed | dispatch #2 implementer resolves one ladder step up; `routing_decision.rule_id == retry-tier-up`; tier persisted in `metadata.routing` |
| 6 | Tier does NOT bump on verify loop | verify fails twice within one dispatch | all implementer/debugger invocations in that dispatch share the same tier |
| 7 | Fable ceiling | `per_model_daily_ceiling_usd[claude-fable-5]` exhausted today; override pins fable | resolution falls back one layer; `rejected` records ceiling; dispatch proceeds |
| 8 | Saturation stamp | simple_task invocation ends at max_turns without completion | `metadata.routing.simple_saturated == true`; next dispatch takes full path; `routing_decision` on that dispatch names the saturation rule |
| 9 | Out-of-band site parity | steward + deep_reviewer invocations for a task with an override covering those roles | their invocation rows honor the override and emit `routing_decision` events |
| 10 | Probe gates fable | availability probe reports fable absent on account max-d | probe artifact lists per-account×model status; ξ's admission step refuses to add fable to `allowed_models` until all pool accounts pass or the failing account is excluded from rotation for fable work |
| 11 | Unknown rule key rejected | `reload_config` with rule containing `plan_vibes: 3` | reload returns ValidationError naming the key; prior rules remain active |
| 12 | Rollup readable | ≥1 invocation per (model×role) pair this window | digest section + dashboard panel show per-(model×role) invocation count, done/blocked/cap-hit rates, $/done |

## Decomposition plan

Phase 1 — substrate (survey Tier 0):

- **α — Retire role-key derivation; make every role config-addressable**
  Modules: orchestrator (workflow.py, config.py, roles.py, tests). Adds
  `simple_task` (and full-name keys generally) to Models/Budgets/Turns/Effort/Timeouts
  submodels; honors-or-deprecates `simple_task_budget_usd`/`simple_task_max_turns`.
  Signal: boundary test 4 (reload flips simple_task model; `applied` disposition +
  invocation row observable). Prereqs: none.
- **β — Model allowlist, validation, fail-fast, availability probe**
  Modules: orchestrator (config.py, new routing.py stub for allowlist), shared
  (cli_invoke invocation-outcome classification). `routing.allowed_models` +
  ValidationError on unknown strings at load/reload; model-not-found API error
  classified terminal (no cross-account retry churn); `orchestrator probe-models` CLI
  subcommand exercising each pool account × allowlist model with a 1-turn probe,
  emitting a committed artifact. Signal: boundary tests 10, 11 + typo'd model reload
  returns a structured error naming the field. Prereqs: none.
- **γ — Routing-decision persistence**
  Modules: orchestrator (event_store.py, workflow.py, harness.py),
  shared (task_metadata.py `routing` typed field). `routing_decision` event per
  invocation + `metadata.routing` mirror (bounded history, tier counter,
  simple_saturated flag storage). Signal: after one dispatch, digest debug section /
  `orchestrator events` query surfaces the routing_decision rows with layer + rule id.
  Prereqs: none (event vocabulary exists).
- **δ — Per-(model×role) outcome rollup**
  Modules: orchestrator (digest.py), dashboard. Rollup from `invocations` ×
  `task_results` × routing_decision events: counts, done/blocked/cap-hit rates, $/done,
  turn-cap saturation rate per role. Steward/triage invocation rows come from **2461**
  (harness-backend T6) — do NOT re-instrument those sites, and the judge already
  records via `_invoke`. In scope here: verify-and-thread `cost_store` for any
  remaining uncosted sites 2461 does not own (module_tagger harness.py:1919,
  unblock_auto) if actually missing. Signal: boundary test 12 (digest section renders;
  dashboard panel renders, incl. steward/triage rows). Prereqs: γ; cross-PRD **2461**.

Phase 2 — resolver (vertical slice):

- **ε — `resolve_route` + policy table, adopted by `_invoke`**
  Modules: orchestrator (routing.py new, workflow.py, config.py RELOADABLE_FIELDS).
  Layered resolution per contract; shipped default rules byte-equivalent (invariant 3
  fixture); `_select_model_for_role` retired into rule
  `rust-large-plan-implementer`; ladder arithmetic + per-model ceilings. Signal:
  boundary tests 1 (via ζ), 3, 5-shape rule matching, 7, 11 — concretely: install a
  test rule via reload, next matching dispatch's invocation row + routing_decision
  show the routed model. Prereqs: α, β, γ.
- **ζ — Typed `metadata.model_overrides` end-to-end**
  Modules: shared (task_metadata.py), fused-memory (server/tools.py submit/update
  guard), orchestrator (resolver layer 1). Shape-validated at submit_task
  (ValidationError on unknown role/non-string), string-validated fail-safe at resolve.
  Signal: boundary tests 1, 2 — submit with malformed overrides rejected; well-formed
  override honored in the invocation row. Prereqs: ε.
- **η — Resolver adoption by out-of-band sites**
  Modules: orchestrator (steward.py, review_checkpoint.py, harness.py module_tagger,
  dry_run_unblock.py). Signal: boundary test 9 (steward/deep_reviewer emit
  routing_decision + honor overrides). Prereqs: ε, ζ.

Phase 3 — fleet rebalancing (survey Tier 1) + plan-shape generalization:

- **θ — Language-agnostic plan-shape rules as fleet defaults**
  Modules: orchestrator (defaults.yaml routing rules; no code). Replace the
  `crates/`-only condition with general `plan_min_steps`/`plan_min_modules` implementer
  upgrade; raise implementer headroom where the rollup shows cap pressure. Applied via
  hot-reload on dark_factory first, then defaults.yaml. Signal: a non-Rust ≥12-step
  plan's implementer invocation resolves opus with `rule_id` naming the new rule
  (routing_decision + invocation row). Prereqs: ε, δ.
- **ι — Architect effort/model decide-and-act (adoption of an eval-revival verdict)**
  Modules: orchestrator (defaults.yaml / per-project yaml config only), committed
  decision record. NO eval-machinery edits (decision 11) — runs the architect
  max-vs-high (and sonnet-architect-for-small-tasks candidate) trial THROUGH the
  revived framework's OFAT→confirm methodology (2478) with architect coverage from
  2475; clear pass → flip `effort.architect`/rule via hot-reload then defaults.yaml;
  marginal → escalate with the report. Signal: committed decision record referencing
  the eval report + either the applied config change (reload disposition) or the filed
  escalation. Prereqs: β, δ; cross-PRD **2475**, **2478**.
- **κ — Haiku decide-and-act pilots: judge, triage, module_tagger**
  Modules: orchestrator (defaults.yaml / config; per-role trial scripts under
  scripts/ or evals/results/ artifacts — no eval-machinery restructuring), committed
  trial reports. Per-role method: **judge** via OFAT single-role substitution
  (`models.judge` override) on the revived framework over its fixtures — on the
  post-2486 verdict-tool contract; **triage** + **module_tagger** via offline
  replay-agreement trials (historical inputs, haiku-vs-sonnet output agreement,
  frontier-adjudicated on disagreements — tier1's D-6 protocol shape, self-contained)
  — triage on the post-2485 contract. Pass → flip that role to haiku on dark_factory
  via hot-reload, watch δ's rollup for a fixed window, then defaults.yaml;
  fail/marginal → report + escalate. Thresholds derive from δ's measured per-role
  baseline (not guessed). Signal: per-role committed report + applied flips visible in
  reload disposition + rollup rows showing haiku invocations for flipped roles.
  Prereqs: δ; cross-PRD **2472**, **2485**, **2486**.
- **λ — simple_task tuning**
  Modules: orchestrator (defaults.yaml / config). Turns 30→50 + budget via α's now-live
  fields; saturation-rate metric (from δ) becomes the tracked indicator. Signal:
  simple_task invocation rows run under the new caps; rollup exposes simple_task
  saturation rate. Prereqs: α, δ.

Phase 4 — adaptive interventions (survey Tier 3):

- **μ — Retry-escalation rung**
  Modules: orchestrator (harness.py dispatch bookkeeping, workflow.py auto-eval,
  escalation server `escalate_model` flag), defaults.yaml `retry-tier-up` rule.
  Tier counter per decision 6; auto-eval redo carries tier+1. Signal: boundary tests
  5, 6 — a task blocked on dispatch #1 shows dispatch #2 implementer one ladder step
  up in routing_decision/invocations. Prereqs: ε, γ.
- **ν — simple_task saturation → full-path stamp**
  Modules: orchestrator (workflow.py simple-task outcome handling, defaults.yaml rule).
  Saturation detection stamps `metadata.routing.simple_saturated`; policy rule routes
  subsequent dispatches full-path. Signal: boundary test 8. Prereqs: ε, γ, λ.
- **ξ — Fable admission**
  Modules: orchestrator (defaults.yaml allowlist+ladder+ceiling; docs). Gated on β's
  probe (all rotation accounts pass for fable, or failing accounts documented+excluded
  for fable work); adds `claude-fable-5` to allowlist/ladder top with
  `per_model_daily_ceiling_usd`; reachable via override + retry final rung only.
  Signal: boundary tests 7, 10 + an override-pinned architect invocation row showing
  claude-fable-5 within ceiling. Prereqs: β, ζ, μ.

Integration gate + companion:

- **σ — Routing integration gate (B+H boundary suite)**
  Modules: orchestrator/tests (workflow test rig, injectable `invoke_fn` seam).
  Drives boundary tests 1–9, 11, 12 end-to-end through the workflow rig with a fake
  agent runner: override wins/falls-through, byte-equivalence fixture, reload rule
  install, tier bump on re-dispatch (and NOT on verify loops), ceiling fallback,
  saturation stamp, out-of-band parity, unknown-rule-key rejection, rollup rows.
  (Test 10 — probe gating — lives in β/ξ, fable-specific.) Signal: the boundary suite
  green against a real workflow-rig run — the C-as-integration-gate leaf closing G2
  for the intermediates. Prereqs: ε, ζ, η, μ, ν, δ.
- **ρ — Docs + operator surface**
  Modules: CLAUDE.md, skills/orchestrate/SKILL.md, plans/. Document model_overrides
  contract, routing config block, probe subcommand, rollup reading. Signal: docs
  committed; `/orchestrate` skill section names the new knobs. Prereqs: ζ, ε.

Leaf/intermediate: α, β, γ, δ, ε, ζ, η, λ, μ, ν are intermediates (each names its
in-batch consumers above; σ is the integration-gate leaf that ropes the resolver-path
intermediates per the G2 escape hatch); θ, ι, κ, ξ, ρ, σ are leaves with the
operator-observable signals stated. G2 hard-check + capability manifest at decompose
time.

## Out of scope

- The backend axis: `backends.<role>` selection (forwarding landed as 2457),
  codex/pi hardening, provider credential pools, per-provider cap semantics, the
  config price table — all owned by `plans/harness-backend-reconnect-pi-prd.md`;
  endpoint-swapped model bundles by eval-revival ν=2479/ξ=2480. The resolver never
  returns a backend.
- Per-(account,model) UsageGate cap states (fable cap = whole-account CAP stands).
- Implementer-saturation → automatic architect decomposition (decomposition machinery,
  not routing; separate PRD if wanted).
- Curation-time LLM routing in the TaskCurator (survey caution: prior automatic
  classifier fired 6× ever; plan-shape + overrides cover the need).
- Architect explicit difficulty-hint plan tool (named follow-up; see Open questions).
- vLLM/self-hosted model re-trials.

## Open questions (surfaced but not decided in this session)

1. **Architect difficulty-hint field**. When the rollup shows where plan-shape routing
   mis-fires, design the plan-tool hint. **Suggested resolution:** file as a follow-up
   PRD/task once θ has ≥4 weeks of routing_decision data. Decide after Phase 3.
2. **Exact haiku pass/fail thresholds per role in κ**. Must derive from δ's measured
   baseline; the marginal band that escalates to a human is set inside κ. Decide during
   κ implementation (basis: reviewer_trial scoring precedent).
3. **`metadata.routing` history depth** (N=5 suggested) and event payload size caps.
   Decide during γ.
4. **Whether λ raises turns to 50 or 40**. Saturation data (28.6% at 30) supports a
   raise; exact value tactical. Decide during λ from δ's saturation curve shape.
