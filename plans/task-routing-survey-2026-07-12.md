# Task Routing Survey — 2026-07-12

Four-agent investigation of the dark-factory task routing system: current mechanics,
full-lifecycle intervention points, historical performance, and multi-model readiness.
Prepared ahead of a planned widening of the model range for tasks and workflow steps.

---

## 1. Executive summary

Routing today is **static and almost entirely decided at submit time**: an author-declared
`complexity='simple'` flag picks the fast path, `task_kind='deterministic'` picks the no-LLM
runner, and everything else takes the full architect path with a fixed per-role model table
(`defaults.yaml`). The only dynamic model selection in the entire system is a hardcoded
sonnet→opus upgrade for implementers on large Rust plans (`workflow.py:7282`).

The good news: the architecture is unusually ready for widening. All per-task LLM calls
funnel through **one chokepoint** (`TaskWorkflow._invoke`, `workflow.py:7351-7359`); per-role
models/effort/budgets/backends are **hot-reloadable**; a **multi-backend dispatcher**
(claude/codex/gemini/vLLM) is fully built; per-invocation **model+cost telemetry already
exists** (`runs.db invocations`); and a **role A/B eval harness** (`reviewer_trial/`) has
already been used once to make a production routing change (5×sonnet panel → 1×opus reviewer).

The bad news: three plumbing defects gate everything (backend param dispatch-dead; the
`simple_task` role invisible to config; zero model-string validation → invalid models cause
retry/failover churn instead of fail-fast), and rich difficulty signals (turn-cap saturation,
verify attempts, review cycles, retry ledgers, dry-run risk labels) are **recorded but never
fed back** into any routing decision. A failed task always retries with the identical
path+model.

Historically the simple path works (½ the escalation rate, ~25% cheaper, far less downstream
churn, done-rate statistically indistinguishable) but over-promises (~half of simple-labeled
tasks fell back to the architect at least once; **28.6% saturate the 30-turn cap** — the
strongest mis-route signal in the data). Spend allocation is inverted: the architect consumes
**47% of all-time spend ($7.1k of $15.2k)** while hitting its turn cap only 0.5% of the time;
the implementer (sonnet) caps out 9.2% of the time.

---

## 2. How routing works today

### 2.1 Decision tree (dispatch → agent running)

1. **Scheduler eligibility** — `scheduler.py:3279` `_eligible_for_dispatch`: status pending →
   not dispatched → deferred-watch → milestone time gate (`:3099`) → requeue cooldown →
   deps (local + external, `:2808`) → dispatch cooldown. PSI load-cap admission at
   `scheduler.py:4562/4642` (deterministic exempt). Scoring: tier base + age/CPM.
2. **Harness slot fork** — `harness.py:5490` `_run_slot`:
   - `task_kind=='deterministic'` (`scheduler.py:1647`) → `DeterministicRunner`
     (`harness.py:5819`) — shell only, **no LLM**.
   - substrate gate probe (`harness.py:5589`) can block before any agent.
   - else construct `TaskWorkflow` (`harness.py:5604`).
3. **Workflow path selection** — `workflow.py:1968-1974`: SIMPLE_TASK fast path iff
   `simple_task_enabled ∧ ¬initial_plan ∧ ¬auto_eval_redo ∧ ¬force_full_path ∧
   is_declared_simple_task` (= `complexity=='simple'` ∧ no hard-blocker regex,
   `triage.py:27-69`). Fallthrough deletes the plan and drops to the architect
   (`workflow.py:1988-1996`). Lever B revalidation-skip (`workflow.py:2862/3166`) skips
   re-planning at zero overlap.
4. **Model resolution** — `workflow.py:7351-7359` (`_invoke`):
   `role_key = role.name.split('_')[0]`; model/budget/turns/effort/timeout/backend via
   `getattr(config.<submodel>, role_key, <role default>)`; then `_select_model_for_role`
   (`:7282`) — the sole dynamic hook (implementer/debugger, ≥3 `crates/` modules ∧ ≥12 plan
   steps → opus).
5. **CLI spawn** — `invoke_with_cap_retry` (`shared/cli_invoke.py:700`) leases an OAuth
   account, then `claude --print --model <str> --effort <str> --max-turns --max-budget-usd`
   (`cli_invoke.py:1225-1251`). **Model string passes verbatim** — `claude-haiku-4-5` or
   `claude-fable-5` would flow through today, unvalidated.

### 2.2 Role → model table (live values, defaults.yaml:158-220)

| Role | Model | Effort | Turns | Budget $ | Cap-hit rate (history) |
|---|---|---|---|---|---|
| architect | opus | max | 100 | 15.0 | 0.5% |
| implementer | sonnet | max | 80 | 10.0 | **9.2%** |
| debugger | sonnet | max | 50 | 5.0 | 7.2% |
| reviewer | opus | high | 30 | 5.0 | 0.1% |
| merger | opus | max | 50 | 5.0 | 6.2% |
| steward | opus | high | 100 | 5.0 | — |
| triage | sonnet | medium | 25 | 2.0 | — |
| module_tagger | sonnet | medium | 30 | 2.0 | — |
| deep_reviewer | opus | max | 100 | 15.0 | — |
| judge | sonnet | medium | 15 | 0.50 | — |
| simple_task | sonnet (**not config-addressable**, roles.py:1168) | 'high' fallback | 30 | 1.50 | **28.6%** |
| unblock_auto | sonnet | high | 50 | 5.0 | — |
| L1 watcher | opus (`watcher_model`, restart-only) | high | 400 | 40/rotation | — |

Fleet-wide: **no live project overrides models/effort/backends** (only the non-daemon
dashboard yaml does). Everything runs defaults.yaml verbatim. Pydantic class defaults
diverge from defaults.yaml (implementer opus-vs-sonnet, reviewer sonnet-vs-opus) — harmless
in production, but the eval runner instantiating `OrchestratorConfig()` sees class defaults.

### 2.3 Non-`_invoke` dispatch sites (must be changed in parallel with any `_invoke` change)

steward (`steward.py:544/575`) + inner triage (`:619`); deep_reviewer
(`review_checkpoint.py:181-190`); module_tagger (`harness.py:1919` — passes no effort/backend);
unblock_auto (`dry_run_unblock.py:316-329`); usage-gate resume probe (**hardcoded
`--model haiku`**, `shared/usage_gate.py:1321`).

---

## 3. Historical performance (tasks.db + runs.db + escalation archive)

Corpus: 2,420 tasks (91.1% done), 908 runs, 12,295 invocations, $15,226 total spend
(2026-04-09 → 2026-07-12). Escalation archive only reliable from mid-June (1,278 records).

### 3.1 Path comparison

| Path | Tasks | Done rate | Disp/task | $/task | Verify attempts (mean) | Review cycles (mean) | Esc/task (id≥1900) |
|---|---|---|---|---|---|---|---|
| Full architect | 1,730 | 92.3% | 1.17 | 5.72 | 0.39 | 0.11 | 1.91 (42% ever esc) |
| Simple | 51 | 88.2% | 1.10 | 4.26 | 0.05 | 0.05 | 1.00 (33%) |
| Deterministic | 23 | (gates pending by design) | 1.00 | ~0 | — | — | 0.52 |

- Simple-path adoption jumped ~10× after the author-declared-complexity PRD (2026-06-23);
  the old Lever-C title-regex classifier fired 6 times ever.
- **But**: ~28/57 simple-labeled tasks also got architect invocations (veto or fallback);
  7/57 blocked/escalated; simple_task saturates its 30-turn cap **28.6%** of runs (vs
  implementer 9.2%, architect 0.5%). Wall-clock per task is no better than the full path.
- `metadata.milestone` and `before_done.kind='predicate'`: **zero usage ever**. Built, never
  exercised.

### 3.2 Spend shape

- By model: opus $9,969 (65%), sonnet $5,257.
- By role: architect **$7,142 (47%)**, implementer $4,595, reviewer $1,993, debugger $548,
  watcher $539, deep_reviewer $235, simple_task $82.
- By phase: plan $5,050 > review $3,464 > execute $3,186 > verify $540 > merge $80.
- Re-dispatch churn: 227 requeued dispatches burned $719; task 2315 took 11 dispatches
  ($28.6). Rebase→re-verify telemetry (471 samples): 7.5h of re-verify wall time.
- Account pool: 1,810 cap_hits, 1,075 failovers — cap-driven failover is routine.

### 3.3 Escalation shape (June–July window)

L2 (human) 217 items, median 2.9h to resolve; design_concern slowest (8.4h median).
Dominant classes are **infra noise, not routing**: `create_worktree leftover-branch` 146,
starvation-watchdog ~200. Dry-run risk labels: only ~15% of at-block proposals were
`low` (auto-unblockable) — the AFK auto-unblock path has narrow coverage.

### 3.4 Prior model experiments (April 2026 bakeoff + reviewer trial)

- Full-workflow Elo bakeoff (~30 configs, 5 tasks): `claude-opus-max` topped Elo;
  **`claude-sonnet-max` was the value pick** (best composite 0.74 @ $4.12/task vs opus 0.67
  @ $8.37). Codex/gemini mid-field; all self-hosted vLLM configs non-viable (composite
  0.09–0.44, frequent zero-done, up to $36/run in pod time).
- `reviewer_trial/`: production 5×sonnet panel F1 0.24 @ $17.69 vs 1×opus generalist F1 0.46
  @ $5.21 (**6.3× F1/$**) — this drove the current 1×opus reviewer config. Effort sweep:
  opus@high beat medium and max on blocking-recall/$ (more effort bought recall, destroyed
  precision). Proof the A/B → config-flip loop works end to end.

---

## 4. Defects and gaps (ordered by how much they block model-widening)

1. **`backends.<role>` is dispatch-dead.** `invoke_with_cap_retry` accepts `backend` but
   never forwards it into `invoke_kwargs` (`cli_invoke.py:715/936/1007`), so `invoke_agent`
   always runs `backend='claude'`. Known deferred bug (`steward.py:556-569`). The entire
   codex/gemini path (`agents/invoke.py:259-554`) is dead code in production. ~One-line
   forwarding fix + credential-separation follow-through.
2. **No model-string validation anywhere.** Invalid/unavailable model → API error →
   classified transient (`cli_invoke.py:467-470`) → steward retries + account failover churn.
   Need an allowlist/startup probe and a fail-fast "model not found" sub-classification.
3. **`simple_task` invisible to config.** `role.name.split('_')[0]` → key `'simple'` misses
   every config submodel; model/budget/turns frozen at role-dataclass defaults;
   `simple_task_budget_usd`/`simple_task_max_turns` (config.py:1903-1904) are **dead fields,
   never read**. Same trap awaits any future underscore-named role.
4. **No per-task or per-step routing surface.** No `metadata.model` contract exists; all
   routing metadata (`complexity`, `force_full_path`, …) rides in `extra='allow'` untyped.
5. **Difficulty signals unused.** verify_attempt, review_cycle, amendment_round,
   WorkflowMetrics, MAX_TURNS exhaustion, retry_ledger, dry-run risk_label, waste_detected,
   rebase_verify_cost — all persisted, none read back by dispatch/`_invoke`. Retry = same
   path, same model, always. The one path-escalation (`_maybe_auto_eval`, `harness.py:5935`)
   only fires for optimistic-path tasks and only forces the full path — never a model bump.
6. **Per-model quota blindness.** UsageGate is per-account; one Opus cap message CAPs the
   whole account even with Sonnet/Haiku headroom (`usage_gate.py:824-854`). Non-Anthropic
   backends would mis-attribute caps to Claude accounts and get a nonsensical
   `claude --model haiku` recovery probe.
7. **Effort tiers unguarded** — passed verbatim; no per-model capability table.
8. **Hardcoded literals**: `'opus'` upgrade `workflow.py:7300`; fallbacks in
   review_checkpoint.py:183-190; `haiku` probe usage_gate.py:1323; eval judge/matcher pins;
   `_MODEL_COSTS` (invoke.py:42-49) covers only 4 non-Anthropic models; ~106 test assertions
   pin sonnet/opus.
9. **Telemetry gaps**: no per-(model×role) outcome rollup (can't detect a weak model
   degrading a role); routing decisions/veto firings not persisted to task metadata; no
   status-transition history in tasks.db; `runs.total_cost_usd` populated in 16/908 runs;
   per-phase duration must be reconstructed from enter/exit events; escalation archive
   rotated away pre-June.

---

## 5. Improvement opportunities across the lifecycle

### Tier 0 — enabling fixes (small, do first)
- Forward `backend` through `invoke_with_cap_retry` → unlocks the already-built
  multi-backend dispatcher.
- Fix role-key derivation (or add `simple_task` fields to the config submodels); wire the
  dead `simple_task_*` config fields. Given the 28.6% cap-saturation, being able to *tune*
  simple_task at all is immediately valuable.
- Model allowlist + fail-fast on model-not-found; per-model effort capability guard.
- Persist the routing decision (path chosen, veto fired, model resolved per role) into task
  metadata / event store — cheap, makes everything later measurable.
- Add per-(model×role) outcome rollup to digest/dashboard (success, block, cap-hit,
  $/outcome) from existing `invocations` rows.

### Tier 1 — config-only rebalancing (evidence already in hand)
- **Architect over-provisioning**: 47% of spend, 0.5% cap rate. Candidates: effort max→high
  (the reviewer sweep showed max can be counterproductive), or sonnet architect for small
  tasks. Validate via evals first.
- **Implementer under-provisioning**: 9.2% cap rate at sonnet/80 turns — raise turns for
  large plans or widen the existing opus-upgrade heuristic beyond Rust.
- **Haiku pilots** on mechanical roles: judge (sonnet/medium/$0.50 — likely fine on haiku),
  triage, module_tagger. Low blast radius, immediate cap-pressure relief on the account pool.
- simple_task turn budget: raise from 30, or better, treat saturation as a signal (Tier 2).

### Tier 2 — per-task/per-step routing surface (the core ask)
- **Typed `metadata.model_overrides`** (per-role map) + validation in TaskMetadata; read in
  `_invoke` between workflow.py:7353 and :7354. Same treatment for effort/backend. Also fixes
  the four non-`_invoke` sites (steward, deep_reviewer, module_tagger, unblock_auto).
- **Generalize `_select_model_for_role`** from the Rust heuristic into a config-driven policy
  table: signals available at plan time = module footprint, plan step count, task priority,
  declared complexity; signals available mid-flight = verify_attempts, review_cycles,
  retry_ledger. Per-phase granularity already exists naturally (each phase = one role
  resolution at `_invoke`).
- **Architect-assisted routing**: the architect's plan is the best complexity estimate the
  system ever produces, and it arrives *before* the expensive implement/verify/review loop.
  Let the plan (step count, module spread, risk notes) set the implementer/debugger
  model+turns per task — this is the cheapest high-quality classifier available, vs adding a
  new LLM classification call at submit/curation time.

### Tier 3 — adaptive lifecycle interventions
- **Escalate model on retry**: attach a model-bump to steward resume / escalation `restart` /
  `_maybe_auto_eval` redo (today auto-eval only sets `force_full_path`). A failed cheap-model
  attempt should re-dispatch on the next tier, not the same one. Nearest existing pattern:
  SIMPLE_TASK's REQUEUED fall-through to the architect (workflow.py:3335-3372).
- **Turn-cap saturation → decompose/promote trigger**: cap exhaustion means "decompose, not
  retry" (task 2169 lesson); make it automatic — saturated simple_task → auto-refile as full
  path (exists) *and* saturated implementer → architect decomposition pass (new).
- **Downgrade chains for cheap-model rollout**: haiku→sonnet→opus fallback rung so a weak
  model degrades to a retry, not an escalation. Prereq for aggressive Haiku adoption.
- **Curation-time routing** (optional, later): the TaskCurator already has LLM + registry
  machinery and a `route_deterministic` precedent; it could stamp suggested
  complexity/model_overrides at create time. Historical caution: the last automatic
  classifier (Lever-C title regex) fired 6 times ever — prefer architect-assisted routing
  first.

### Tier 4 — wider backend range
- After Tier-0 backend fix: codex/gemini per-role trials via the eval harness. Real work
  remains in the UsageGate (per-provider credential pools, per-(account,model) cap states,
  provider-appropriate resume probes) before production use.
- vLLM/self-hosted via `ANTHROPIC_BASE_URL` bridge is production-wired already
  (cli_invoke.py:1303-1314) but April's bakeoff found nothing self-hosted viable; re-test
  only when a materially better open model ships. Note env_overrides currently reach only
  implementer/debugger.

---

## 6. Recommended rollout mechanics for new models

1. Offline A/B via `reviewer_trial/` (role-shaped; ~90% ready — needs a `backend` field on
   `ReviewerSpec`) or full-workflow eval (`evals/runner.py`; needs `build_eval_orch_config`
   un-hardcoded so configs can target roles other than implementer/debugger, ~10 lines).
2. Config flip on **one** project's orchestrator.yaml via green-tier hot-reload
   (`models.*`/`effort.*`/`backends.*` all reloadable, read-at-spawn).
3. Watch the new per-(model×role) rollup (Tier 0) + block-rate + cap-hit rate for a window;
   then fleet-wide defaults.yaml change.
4. Keep the judge/eval instruments pinned to concrete model IDs so the measuring stick
   doesn't move when aliases shift.

---

## 7. Adjacent findings (not routing, worth owning separately)

- Escalation signal quality is drowned by infra noise: `create_worktree leftover-branch`
  (146) + starvation-watchdog (~200) dominate the archive. Fixing that bug class buys more
  human-attention relief than any routing change.
- Milestone + predicate mechanisms: zero lifetime usage — decide to promote or stop
  maintaining.
- `runs.total_cost_usd` rollup is dead (16/908) — either fix or drop the column.
- Pydantic-vs-defaults.yaml default divergence (implementer/reviewer) is a latent eval-vs-prod
  inconsistency.

---

*Sources: four-agent survey (routing code map; lifecycle/intervention map; runs.db+tasks.db+
escalation-archive mining; multi-model readiness audit), 2026-07-12. Quantitative queries run
read-only against `.taskmaster/tasks/tasks.db` (user_version=4) and
`data/orchestrator/runs.db`.*
