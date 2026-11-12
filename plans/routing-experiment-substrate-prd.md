# PRD: Routing experiment substrate and the first measured routing changes

**Status:** active — authored 2026-11-03 from Leo's rulings on
`plans/portfolio-review-2026-11-01/followups/model-routing.md` §13.5 (answers 2, 3 and 6).
**Project:** dark_factory (the resolver and the curator run in every orchestrator and in
fused-memory, so every hosted project is covered by the same code). **Approach:** bare B with one
contract: the split function has three consumers that must agree bit-for-bit, so it gets a
signature, invariants and a two-sided boundary test; everything else is a shallow DAG.

> **Code anchors** verified against main `d5f9c77e50` (2026-11-03). Main moves fast — cite-by-
> symbol; re-locate at implementation time.

## Goal

Make routing changes measurable, then make the two routing changes the evidence supports and
measure them:

1. **Substrate.** A `split` condition in the routing rule vocabulary, keyed on a salted hash of the
   task id, so a rule can apply to one arm of a concurrent fleet-wide split; plan-shape fields on
   every `routing_decision` event so a task's plan size is a recorded covariate; and one read-only
   readout script that computes per-task, per-arm metrics across every project ledger, runs an
   A/A check on an unchanged window, and prints the null distribution the revert bound is read
   against.
2. **Experiment 1 (routing).** The implementer's `large-plan-modules` threshold moves from 3 to 4
   modules on one arm of a two-arm split, fleet-wide, until a pre-registered sample or 28 days.
3. **Experiment 2 (declaration).** The curator stamps a `complexity=simple` hint from a cheap tier
   judgement on every created task it curates, and applies it as a real declaration on one arm
   of a second, independently salted split.
4. **Readout gate.** One dated human gate reads both experiments against their A/A nulls and rules
   keep, revert or extend.

**User-observable surface (G2 top signal):** the readout gate's born-at-L2 escalation carries the
readout script's per-arm table for both experiments, each metric beside its A/A 95th and 99th
percentiles, and Leo resolves it with a ruling. Along the way: `routing_decision` events carry
`plan_steps` and `plan_modules`; `scripts/routing_experiment_readout.py --aa` prints an
imbalance table on an unchanged window and exits non-zero when the split is skewed.

**Named consumers (G1):**

- The `split` condition: consumed by experiment 1's rules in `defaults.yaml` (leaf γ) and by
  experiment 2's arm selection in the curator (leaf ε).
- `split_bucket` (the function): consumed by the resolver (α), the curator (ε) and the readout
  script (β). That is the seam under contract.
- Plan-shape fields on `routing_decision`: consumed by the readout script (β) as the plan-size
  covariate and to identify experiment 1's affected subset.
- The readout script: consumed by the arm-on leaf's A/A gate (δ) and by the readout gate (η).
- The curator hint: consumed by the fast path (`workflow.py::_should_run_simple_task` reads
  `metadata.complexity`) on the treated arm, and by the readout (η) on both arms.

## Background

- Resolver: `orchestrator/src/orchestrator/routing.py::resolve_route` with the closed
  `config.py::RuleMatch` vocabulary (`extra='forbid'`); first-match-wins; a project's
  `routing.rules:` list replaces the fleet list (`config.py::_deep_merge`), and today both
  dark-factory's and reify's yaml restate the four stock rules. `RouteInputs` carries `task_id`
  and `plan_shape = PlanShape(step_count, module_paths)`; the `routing_decision` event records
  role, model, effort, budget, turns, source_layer, rule_id, rejected, routing_tier and an
  `inputs_digest`, but not the plan shape itself.
- Evidence (window 2026-10-02..11-02, seven ledgers; `followups/model-routing.md` §2–§4, §13):
  the implementer runs opus on 82% of decisions, 48% of them lifted from configured sonnet by the
  two large-plan rules; `large-plan-modules` (≥3) fires on 62% of decisions; sonnet on small plans
  is 99% done at $2.26 per invocation. Filing text cannot predict plan size (experiment 0), but a
  cheap tier judgement orders tasks by cost and failure, and three quarters of its "mechanical"
  bucket is undeclared simple. The declared fast path is 96% done at 0.2× cost with a 16%
  architect fallback.
- Measurement (seat N, `followups/model-routing-seats/N-null-model.md`): a single-project
  before/after canary cannot measure anything (false-revert 20–50%; within-month drift of +5pp on
  the blocked rate with no intervention). Only a concurrent split with an A/A check can; task-id
  parity is skewed (odd ids 11% cheaper in dark-factory), so the split must be a salted hash and
  must be checked on an unchanged window before arming.
- Where the contrast lives: dark-factory's implementer is opus by config (`models.implementer:
  opus`), so experiment 1 changes nothing there; reify restates the rule list and its scheduler is
  halted (2026-11-02 ruling). The treated population is the five client/greenfield projects,
  about 2,000 implementer-reaching tasks per month between them, so roughly 500 per arm per
  fortnight; the affected subset (plans with exactly 3 modules and fewer than 12 steps) is
  unknown until α records `plan_modules`, which is why the sample target is set on the affected
  subset and the gate is dated at 28 days with an interim read at 14.
- The curator (`fused-memory/src/fused_memory/middleware/task_curator.py::CURATOR_OUTPUT_SCHEMA`,
  sonnet) already returns an optional `rewritten_task` that
  `task_interceptor.py` applies on create via `CandidateTask.from_rewritten_task`; its schema has
  no complexity field. `planning_mode` submissions bypass the curator, so PRD-decomposed batches
  are outside experiment 2 by construction; `agent-followup` filings (58% of tasks) are inside it.
- Rulings that bind: COMMON.md ruling 2 (verify CPU is the scarce resource: verify minutes per
  landed task is the guard metric); Leo 2026-11-03 answers 1–6 (strongest model at PRD and
  architect; substrate first; implementer threshold as the first live change; no fable rung; the
  registry later; pursue the declaration nudge).

## Sketch of approach

### The split (α)

`shared/src/shared/routing_split.py::split_bucket(salt: str, task_id: str, buckets: int) -> int`
returns `int.from_bytes(sha256(f"{salt}\x00{task_id}").digest()[:8], 'big') % buckets`. It lives
in `shared/` because fused-memory (the curator) and the orchestrator (the resolver, the readout)
both import it and neither imports the other. One home (INV-9).

`RuleMatch` gains `split: SplitMatch | None` with `SplitMatch(salt: str, buckets: int ≥ 2,
bucket: int, 0 ≤ bucket < buckets)`, validated at load and reload (`extra='forbid'` as the rest of
the vocabulary; a malformed split fails the config load, never matches silently: INV-11).
`_rule_matches` adds one AND-condition: `split_bucket(salt, inputs.task_id, buckets) == bucket`.
A rule with a split condition is otherwise an ordinary rule: first-match-wins and ladder
semantics are unchanged.

`_record_routing_decision` and `routing_dispatch.py::resolve_and_record_route` add two fields to
the event payload and the `metadata.routing` mirror: `plan_steps` (int or null) and
`plan_modules` (int or null), from the same `PlanShape` the resolver saw. The arm is NOT recorded
on the event: it is a pure function of (salt, task_id) and the readout recomputes it, so there is
exactly one definition of the arm and no event-to-rule drift to reconcile.

### The readout (β)

`scripts/routing_experiment_readout.py`, read-only (`file:…?mode=ro`), over every project root
named in `DASHBOARD_KNOWN_PROJECT_ROOTS` or a `--roots` list. Per task in the window: project,
bucket under `--salt/--buckets`, implementer model set, `plan_steps`/`plan_modules` from the
implementer's `routing_decision`, whether a large-plan rule fired, `metadata.complexity` and the
curator hint, final `task_results` outcome, blocked-or-escalated, max review cycles, max execute
iterations, total invocation cost, workflow-verify minutes (`workflow_verify.data.duration_ms`,
present since 2026-10-29), merge-verify minutes, reviewer blocking count. Per arm and per
experiment it prints the metric means with the A/A null (4,000 resamples of the same window
split at random, seed given) 95th and 99th percentiles beside each difference, and the minimum
detectable effect at the observed n. `--aa` mode: the same table on a window with no treatment,
exiting non-zero if any headline metric's observed arm difference exceeds its own resampled 99th
percentile; that is the check δ runs before arming. Every table carries the rule it was
computed under. The seat N scripts in `/tmp/exp0n/` are the worked prototype of the resampling.

### Experiment 1: rules (γ, δ)

In `defaults.yaml` the rule table becomes, in order: `large-plan-steps` (unchanged, both arms);
`large-plan-modules-4` (`plan_min_modules: 4`, `split: {salt: routing-exp-1, buckets: 2,
bucket: 1}` → opus, 120 turns); `large-plan-modules` (`plan_min_modules: 3`, `split: {…,
bucket: 0}` → opus, 120 turns); then the unchanged `simple-saturated-full-path` and
`retry-tier-up`. Arm 1 plans with exactly 3 modules and fewer than 12 steps fall through to the
role's configured model (sonnet in the five treated projects). Dark-factory's yaml restates the
list with the same two rules so its decisions carry the same rule ids (it contributes a null
contrast, since its implementer is opus by config). Reify's list is left as is.

Pre-registration (written into γ's task text and η's gate text before δ fires): primary metric
blocked-or-escalated on the affected subset; guard metric workflow-verify minutes per landed task
per arm (ruling 2); secondaries review cycles ≥1, execute iterations, cost per landed task,
reviewer blocking count. Stop and revert rule: the guard or primary exceeds the A/A 99th
percentile at an interim read with n ≥ 100 affected tasks per arm; revert is deleting the split
from the two rules and reloading. Sample target: 400 affected-subset tasks per arm or 28 days,
whichever first. δ arms it: a deterministic leaf whose `before_done` script runs the readout in
`--aa` mode on the trailing 30 days with the experiment salt (must exit 0), then issues
`reload_config` to every running unit and records the flip timestamp in its result.

### Experiment 2: the curator hint (ε)

`CURATOR_OUTPUT_SCHEMA` and the batch schema gain `complexity_hint: "simple" | null`; the prompt
gains the fast path's own criteria (`docs/task-authoring.md` §4: a single coherent edit, no new
abstraction, no cross-module design, target files nameable) and asks for the hint on every
`create` and `combine` decision. The interceptor, on create: always stamps
`metadata.x_curator_complexity_hint` (the counterfactual label on both arms); on the treated arm
it also sets `metadata.complexity = "simple"` iff the hint is `simple`, the task carries a
non-empty files list, and the submitter did not set `complexity` itself (a filer's own
declaration always wins). The arm key is NOT the task id, which does not exist until the store
inserts the row: it is the candidate key,
`fused_memory.middleware.candidate_key.compute_candidate_key(title, files)` over the FINAL
(post-rewrite) title and files, which the store persists in `tasks.candidate_key`, so the readout
recomputes the arm from the stored column: `split_bucket("routing-exp-2", candidate_key, 2) == 1`.
A task whose key is uncomputable (empty title) records the hint only. The workflow's existing
hard-blocker veto and 16% architect fallback remain the safety net; nothing else in the fast path
changes. Readout: per arm, share declared simple, simple-path share, fallback-to-architect share,
done rate, cost, blocked-or-escalated, with the untreated arm's hint as the counterfactual.

### Readout gate (η)

A deterministic `always_escalates` pure gate with a delayed milestone anchored on δ reaching done
plus 28 days. Its text names the readout command for both experiments, the pre-registered
metrics and stop rules, and the three rulings available per experiment: keep (threshold 4 becomes
the stock rule / the hint applies fleet-wide without a split), revert, extend. An interim read at
14 days is Leo's option, not a second gate.

### Documentation (ζ)

`OPERATIONS.md` §7: the `split` row in the condition table; the "Adaptive-routing substrate (not
yet active)" subsection corrected on its four stale counts (the retry-tier bump, the saturation
stamp, fable admission and the model×role rollup are live). `docs/task-authoring.md` §4: one
paragraph on the curator hint and that a filer's own `complexity` always wins. The filing skills
that already name the fast path (`skills/prd/references/decompose-mode.md`, `skills/review-all`,
`skills/hotspot-survey`) get a one-line pointer, not a restatement (INV-9).

## Resolved design decisions

1. **Arm is a pure function, not a recorded field.** `split_bucket` has one home in `shared/`;
   the resolver, the curator and the readout call it. Recording the arm on the event would be a
   second home for the same fact and would need its own reconciliation (INV-9).
2. **Salted sha256, not task-id parity.** Seat N measured parity skew; a salt per experiment also
   makes the two experiments independent splits.
3. **Plan shape goes on the event.** `plan_steps`/`plan_modules` are the covariate Leo asked for
   and the only way to identify experiment 1's affected subset; the rule id alone cannot
   distinguish a 3-module plan from a 7-module one.
4. **Readout as a script over all ledgers, not a digest panel.** The digest and dashboard rollup
   is per project and per invocation; the experiment is fleet-wide and per task. A digest change
   is deferred (open question 1).
5. **Experiment 1 by rule ordering, not a new threshold key.** Two split rules at thresholds 4 and
   3 express the arms in the existing vocabulary; no resolver semantics change beyond the split
   condition.
6. **The curator applies the declaration on the treated arm; it never routes.** The declaration
   is the existing contract with its existing fallback; a filer's explicit `complexity` wins; the
   untreated arm keeps the hint as a label so the readout has a counterfactual. The arm key is
   the stored `candidate_key`, because the task id does not exist when the decision is applied
   (verified: `compute_candidate_key` hashes normalised title+files and the store persists it).
7. **Pre-registration lives in the task texts (γ, η), not only in the PRD**, so the gate's
   escalation text carries the stop rule the ruling is made against.
8. **Treated population for experiment 1 is the five client/greenfield projects**; dark-factory
   runs the same rules as a null contrast and reify is left out.
9. **Out of scope and deliberately so:** any architect model or effort change, the model
   registry, fable on the ladder, a tier change for any role other than the implementer and
   debugger on the arm rule (debugger follows the same rules as today).

## Pre-conditions for activating

- Main at or after the architect-effort carry (`t3/default-architect-effort-xhigh`, merge
  request mr-209e3863) is not required; the two are independent.
- `workflow_verify.data.duration_ms` exists on main since 2026-10-29 (verified in the ledger,
  508 events in dark-factory since then); the guard metric reads it.
- `reload_config` re-reads `defaults.yaml` through `config.py::_load_defaults` on every load
  (verified), so γ's rule edit goes live per unit on reload without a restart.

## Cross-PRD relationship

| PRD | direction | mechanism | owner |
|---|---|---|---|
| `plans/adaptive-model-routing-prd.md` | extends | `RuleMatch` vocabulary (+`split`), `routing_decision` payload (+plan shape) | this PRD (α) |
| `plans/live-shadow-eval-prd.md` | none | the shadow eval samples tasks by its own `task_sampler`; it does not read `routing.rules` and is unaffected by a split | — |
| `plans/config-hot-reload-prd.md` | consumes | `routing.*` stays green tier; a `split` typo fails reload with a structured error as today | existing |
| `docs/quality-findings-contract.md` | none | no findings are remediated here | — |

## Contract: `split_bucket`

```python
# shared/src/shared/routing_split.py
def split_bucket(salt: str, task_id: str, buckets: int) -> int:
    """Deterministic arm for (salt, task_id). Pure; no I/O; stable across processes and hosts."""
```

Invariants: `buckets >= 2`; `0 <= result < buckets`; `task_id` is the task's id as a string
exactly as `RouteInputs.task_id` carries it (digits, no project prefix) for experiment 1, and the
16-char `candidate_key` for experiment 2 (the second argument is any stable string key; the
caller names which); the same (salt, key) always yields the same bucket; changing the salt
re-shuffles arms. Error semantics: `buckets < 2`
raises `ValueError` at the call site, and `SplitMatch` validation makes that unreachable from
config.

Boundary tests (α, both sides of the seam):

| scenario | preconditions | postconditions |
|---|---|---|
| resolver and readout agree | 1,000 synthetic task ids, salt "t", buckets 2 | `_rule_matches` with a split rule selects exactly the ids for which `split_bucket` returns the rule's bucket |
| arms are balanced | 10,000 ids, buckets 2 | bucket-1 share within 48–52% |
| salt independence | same ids, salts "a" and "b" | agreement between the two bucketings within 48–52% |
| malformed split rejected at load | `split: {salt: x, buckets: 1, bucket: 0}` and `{…, buckets: 2, bucket: 2}` | config load raises a structured validation error naming the field |
| first-match order preserved | the experiment-1 rule table, a 3-module/5-step plan | bucket-0 task resolves `large-plan-modules`; bucket-1 task resolves no rule (config model) |
| plan shape recorded | an implementer invocation with a 5-step, 3-module plan | the `routing_decision` event carries `plan_steps: 5, plan_modules: 3`; an architect invocation carries nulls |

## Decomposition plan

Labels are PRD-internal; ids are assigned at decompose. Sizing per the overlay's bands.

- **α — split condition, plan-shape fields, `split_bucket`** (normal; `shared/`, `orchestrator/`
  routing.py, config.py, workflow.py `_record_routing_decision`, routing_dispatch.py, tests incl.
  the boundary table above; `OPERATIONS.md` §7 condition-table row only). Signal: the boundary
  tests pass and a live `routing_decision` event carries `plan_steps`/`plan_modules`. Prereqs:
  none.
- **β — readout and A/A script** (normal; `scripts/routing_experiment_readout.py`, a test with a
  fixture ledger exercising both modes and the non-zero exit on skew). Signal: `--aa` on the
  trailing 30 days with salt `routing-exp-1` prints the imbalance table and exits 0; a fixture
  with a forced skew exits non-zero. Prereqs: α.
- **γ — experiment-1 rule table** (normal, `complexity=simple`; `defaults.yaml`,
  `dark-factory-orchestrator.yaml` rules block, `test_routing_byte_equivalence.py` or its
  successor updated to the new table, pre-registration text in the task). Signal: the config
  tests pass with the two split rules in the stated order; `resolve_route` on a 3-module plan
  yields opus for bucket 0 and the config model for bucket 1. Prereqs: α.
- **δ — arm experiment 1** (deterministic; `before_done` = `scripts/routing_experiment_arm.sh`,
  which runs β `--aa` with the salt and reloads every running unit, recording the flip time).
  Signal: every unit's `reload_config` reports `routing.rules` applied; the first
  `routing_decision` with `rule_id: large-plan-modules-4` appears in a treated ledger. Prereqs: β,
  γ landed and deployed (the before_done checks that main carries γ's rule ids).
- **ε — curator complexity hint and treated-arm declaration** (normal; fused-memory
  task_curator.py schema and prompt, curator_types.py, task_interceptor.py create path keyed on
  `compute_candidate_key` of the final title+files, shared import of `split_bucket`, tests on
  both arms, on the filer-wins rule and on an uncomputable key). Signal: a curated
  create on the treated arm with a `simple` hint and files lands with `metadata.complexity:
  simple`; the same candidate on the untreated arm lands with only the hint stamped. Prereqs: α.
- **ζ — documentation** (normal, `complexity=simple`; `OPERATIONS.md` §7 stale subsection,
  `docs/task-authoring.md` §4, skill pointers). Signal: the four stale claims are gone and the
  hint paragraph exists. Prereqs: α (vocabulary row lands with α; this leaf is the rest).
- **η — readout gate** (deterministic, `always_escalates`, delayed milestone anchored on δ + 28
  days). Signal: the born-at-L2 escalation carries both readouts and Leo resolves it with
  keep/revert/extend per experiment. Prereqs: δ, ε, β.

Same-file serialisation: α and ζ both edit `OPERATIONS.md` (α the table row, ζ the subsection);
ζ depends on α. γ and α both touch orchestrator tests; γ depends on α.

## Out of scope

Architect model or effort changes (ruled separately); the model registry (build when a
production flip is ruled); fable or an effort rung on the ladder (no population of tasks opus
could not plan exists: seat U); a digest or dashboard panel for experiments (open question 1);
raising the five projects' architect turn cap from 100 (seat U's cheaper probe for the 17 no-plan
failures; a separate one-line ruling).

## Open questions (tactical)

1. **Digest integration of the per-task readout.** Suggested resolution: after η rules, if the
   split mechanism stays, add a per-task rollup to `digest.py` keyed by salt. Decide at η.
2. **Interim read mechanics.** Suggested resolution: Leo runs the readout by hand at 14 days; no
   second gate. Decide during δ.
3. **Whether the debugger should follow the arm rule.** Suggested resolution: yes, as today the
   two roles share the rule; its decisions are 5% of the implementer's and do not move the
   readout. Decide during γ.
