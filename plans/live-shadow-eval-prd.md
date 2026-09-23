# PRD: Live paired shadow eval — measuring candidate models on the current codebase

**Status:** active — authored 2026-09-10 (interactive design session with Leo; the four
shape/record/slate/supersession decisions and the architect-consequence shape were ruled in
conversation and are honored here, not re-litigated). **Project:** dark_factory (the
mechanism runs in every orchestrator process, so reify is covered by the same code).
**Approach:** **B+H** — G5 heuristic hit on three axes: cross-module blast radius ≥ 3
(`orchestrator/harness.py`, `orchestrator/evals/*`, `orchestrator/config.py`,
`orchestrator/cli.py`, `orchestrator/run_store.py`), a **load-bearing seam** (the harness
dispatch path and the merge-lane landing event), and **two cross-PRD consumers** (the fable
η gate and the adaptive-routing flip).

> **Code anchors** verified against main `32d0f6d1a7` (`2026-09-10`), re-measured against
> main at decompose on `2026-09-11`. Main moves fast — cite-by-symbol; re-locate at
> implementation time.

> **Corrections.** Eight premises written here on 2026-09-10 were measured false or unbuilt
> during the decompose gate walk and are **fixed in place** throughout this document, tagged
> **[S1]**–**[S8]** where they bite. What was originally claimed, what was measured, and which
> leaf absorbed each resolution is recorded once — in
> `plans/live-shadow-eval-prd.capability-manifest.md` §"Substrate corrections made at
> decompose". This document states only what is true (INV-9 `one-fact-one-home`).

## Goal

Replace the frozen fixture corpus as the **capability instrument** with a live, paired
shadow evaluation: when the scheduler dispatches a sampled production task, a candidate
model is run in the role under test against the **same task at the same base commit**, in a
throwaway worktree that never enters the merge lane, and its outcome is scored against what
production actually landed.

**User-observable surface (G2 top signal):** one CLI invocation —

```
orchestrator eval-shadow report [--shape implementer|architect|architect-consequence|end-to-end] [--candidate NAME]
```

— prints, per candidate and shape, the **paired** delta against the incumbent over the
cells settled so far: pair count, the shape's primary outcome with a bootstrap CI, cost per
usable outcome, the decline split (`terminal_kind`), and the cap-excluded / unreferenced /
expired counts. Every line traces to rows in one record store; nothing is derived from
transcripts or logs.

**Named consumers (G1):**

1. **esc-3637-1 / task 3637** (fable-trial-v2 η, the admission re-ruling gate, open since
   2026-08-30; the task itself is `blocked` on task 3636). Leo ruled 2026-09-10 that *this PRD is the redesigned eval set* the
   v2 decision record deferred to. The gate's ruling consumes this PRD's report over the
   `architect-consequence` shape for `architect-fable-max` vs `architect-opus-max`.
2. **`plans/adaptive-model-routing-prd.md`** — any production routing change stays a
   separate deterministic-deploy task filed on a favourable report (this PRD measures; it
   never flips config).
3. **`plans/eval-framework-revival-prd.md` decision 9's *confirm* stage** — the
   `end-to-end` shape is its live analogue; the offline `eval-confirm` driver is no longer
   the path to a production change.

## Background — why the frozen corpus cannot answer the question

Measured 2026-09-09 against main (fixture counts re-counted 2026-09-11):

| Corpus | Fixtures | With frozen plan | Base age | Commits behind HEAD |
|---|---|---|---|---|
| `evals/tasks/` | 22 | **6** (five are April fixtures) | Apr–Jul 2026 | 7k–65k |
| `evals/tasks_hard_v2/` | 42 | 3 | Apr–Jul 2026 | 6k–65k |

- The **implementer** screen needs a frozen plan; five of the six plan-bearing fixtures are
  the April tasks (`df_task_12/13/18`, `reify_task_12/27`) — a codebase 59k–65k commits
  ago. Main moves ~400 commits/day; any frozen fixture is weeks from irrelevance.
- **Historical replay is incoherent by construction.** The eval worktree shares the live
  object DB, refs, task DB and escalation queue, so an honest architect sees the task
  already landed and correctly declines — 47 of 53 cells in fable-trial-v2 tranche 1
  (`plans/fable-architect-trial-v2-decision-2026-08-30.md` §8). Task 4844 (which absorbed
  4758 and 4765) tries to patch the replay frame; a live eval makes the frame moot.
- The non-Claude candidate bundles (`evals/configs.py::claude_endpoint_candidates`,
  `codex_pi_candidates`) have **never produced a result**: `eval-ofat` hardcodes
  `ofat_candidates()` and excludes them, and the slate is stale — the retired
  `codex-gpt54*` / `gemini-*` entries in `EVAL_CONFIGS` date to **2026-03-19**
  (`d3b14de8107`), and the endpoint bundles still name `MiniMax-M2.5`, `glm-5.2`,
  `deepseek-v4`, `kimi-latest`.
- **The v1 fable verdict tied on plan quality** (0.919 vs 0.909, six near-ceiling
  fixtures); the judge scores a plan against the incumbent's landed diff, so a better plan
  that takes a different route is penalised. Leo's live hypothesis — *Opus can plan; Fable
  plans better* — needs an instrument that measures a plan by its **consequences**, not its
  resemblance.

## Premise (G6)

Re-measured against main at decompose, 2026-09-11. Corrected claims carry their **[S*]** tag;
see the Corrections note in the header for where the provenance lives.

- `run_eval`, `run_architect_eval`, `run_end_to_end` (`evals/runner.py`) accept a fixture path
  and are driven end-to-end by the July revival. **Verified.** Their fixture contract is the
  `load_task` dict, and it is **wider** than the keys first listed here: beyond `id`,
  `project_root`, `pre_task_commit`, `task_definition`, `verify_commands`, `plan`,
  `reference.post_task_commit` and `timeout_minutes`, consumers also read `setup_commands`,
  `modules`, `name`, `max_execute_iterations`, `max_review_cycles`,
  `judge_after_each_iteration`, `max_architect_turns` and `adversarial`. `run_eval` raises
  `ValueError` on a falsy `plan`.
- Production persists the accepted plan durably as `plan.json` under the task's meta root
  (`artifacts.py::TaskArtifacts.write_plan`, writer call site `mcp/plan_tools.py::_create_plan`).
  **[S8]** That root comes from `TaskArtifacts.meta_root_for(worktree_base, worktree_name)`, so
  the real path is `<worktree_base>/.task-meta/<worktree_name>/plan.json` — keyed by **worktree
  name**, not by task id.
- The task record carries `metadata.branch_base_sha` (the dispatch base), written by
  `workflow.py::TaskWorkflow._setup_worktree_and_artifacts` immediately after worktree creation
  — and written **soft-failing**, so a cell must treat its absence as a skip with a reason
  rather than guess HEAD. **Verified.**
- `EventType.phase_enter` / `phase_exit` carry `(task_id, phase)` and are emitted from exactly
  one site, `workflow.py::TaskWorkflow._enter_phase`. `EventType.merge_finalized` carries
  `branch`, `state`, `merge_sha`, `snapshot_tip`, `superseded_by` (plus `request_id`,
  `generation`, `reason`, `landed_via_chain`) at its emit site — the `_on_finalized` closure in
  `merge_queue.py::enqueue_merge_request`. `EventStore.latest_merge_finalized(branch=)` and
  `fetch_events_by_type` exist. **Verified.**
  **[S4]** But `merge_finalized` has **no consumer inside the orchestrator process outside the
  merge worker**: `TaskWorkflow` takes its merge outcome from the awaited `MergeRequest.result`
  future, not from the event. The settle hook is therefore a further `add_done_callback` on that
  future, where two independent callbacks already coexist. Relatedly,
  `_await_cancellable`'s `on_soft_cancel` is the **only** pre-existing callback seam on the
  harness — `EventStore` has no subscribe API — so the phase hook is the first observer.
- `evals/snapshots.py::create_eval_worktree` places a worktree at
  `<project_root.parent>/<project_root.name>-eval-worktrees/<id>/run-<8hex>/` — a **sibling** of
  `project_root`, outside `.worktrees/`. **Verified.**
  **[S5]** It uses `git worktree add --detach`, so it creates **no ref at all**. A cell has no
  branch and nothing it could enqueue. The four worktree reapers
  (`git_ops.py::prune_stale_merge_worktrees`, `::reap_interactive_worktrees`,
  `harness.py::_reap_orphan_worktrees`, `::_run_interactive_worktree_reaper_pass`) all filter on
  the worktree's parent being `worktree_base` (`GitConfig.worktree_dir`, default `.worktrees`),
  so the eval root is already structurally invisible to them and **no exclusion needs adding**.
- `EvalMetrics` already carries `terminal_kind` (task 4760), `cap_tainted`, `role_under_test`,
  `cost_source`; `EvalResult` is persisted by `save_result` as
  `<task>__<config>__<run_id>.json`. Both are plain dataclasses with every field defaulted, and
  every read-back site whitelists via `__dataclass_fields__`, so additive fields
  (`shadow_cell_id`, `reference_kind`) are safe. **Verified.**
- **[S1]** Production's paired metrics are **not readable today**, and the two sources have
  different owners. `task_results` belongs to `orchestrator/run_store.py::RunStore`, whose only
  accessor is `get_task_cost` (an aggregate cost float — there is no outcome accessor). The
  *invocation ledger* is **`shared/src/shared/cost_store.py::CostStore`**'s `invocations` table
  — a separate class in a separate package that merely shares the `runs.db` **file** with
  `RunStore`/`EventStore` — and it exposes only window aggregates; per-task filtering is done
  today by callers hand-writing SQL (`harness.py::Harness._auto_eval_budget_used_24h`). Leaf α
  builds both accessors, and retires that hand-written site.
- The dispatch load gate is `scheduler.py::_phase_psi_gate` over `shared.psi`. **Verified** —
  noting `saturated(cfg)` is a **method on the frozen `PsiSample` dataclass**, not a module-level
  function, and returns `False` whenever `read_ok` is `False` (a deliberate fail-open).
- A usage-cap hit is the **boolean return** of
  `shared/src/shared/usage_gate.py::UsageGate.detect_cap_hit` — not an exception, and not a field
  on a result. `SessionBudgetExhausted` is a different mechanism (local session budget) and must
  not be conflated with it. `evals/runner.py::_build_eval_usage_gate` returns `None` fail-open.
  **Verified.** (`orchestrator/usage_gate.py` is a re-export shim; cite the `shared` path.)
- The EXECUTE phase already re-invokes the implementer while `artifacts.get_pending_steps()` is
  non-empty, bounded by `config.max_execute_iterations` (`workflow.py::_execute_iterations`,
  whose cap check subtracts `metrics.progress_resume_total` — a scaled bound must too); the brief
  is assembled by `briefing.py::build_implementer_prompt` from the plan's `pending`/`done` steps
  and today ends "Execute the next pending steps in TDD order … Stop at a logical boundary";
  `judge_after_each_iteration` exists; `EvalConfig` is a plain dataclass that
  `build_eval_orch_config` maps onto per-role config; `get_config_by_name` is a linear scan
  returning `None` on a miss. A one-step variant is therefore a brief branch plus bound scaling,
  not a workflow change. **Verified.** No prior design or task for step-wise execution exists in
  plans, docs, briefs, tasks or memory (searched 2026-09-10).
  **[S3]** But the iteration log's `steps_completed` is a **list**, computed as a set difference
  of before/after `done` step ids — zero, one or many per iteration. "Exactly one step per
  invocation" is what leaf ν *delivers*, never a property to assert of today's mechanism.
- **[S6]** **Sandboxing does not reach eval invocations.** `evals/runner.py`, `evals/compare.py`
  and `evals/judge.py` all call `agents/invoke.py::invoke_agent` **without** `sandbox_modules`,
  and the entire wrap path inside `_invoke_claude_with_sandbox` is gated on that argument being
  non-`None`; `EVAL_PROFILE` carries no sandbox key. Landlock wrapping for shadow implementer
  legs is plumbing leaf δ must build, not a capability to inherit.
- **[S2]** **A new config block is not green-tier by imitation.**
  `OrchestratorConfig.auto_eval_redo_budget_usd` — the budget pattern this PRD copies — is **not**
  in `config.py::RELOADABLE_FIELDS`; it is restart-tier. Hot reload requires an explicit
  `_submodel_leaf_paths('shadow_eval', ShadowEvalConfig)` registration, which leaf α does. (A new
  field on an *already-registered* submodel needs no further edit.)
- **[S7]** **The candidate-slate tests pin wiring, not values.**
  `test_eval_candidate_bundles.py` and `test_eval_codex_pi_bundles.py` assert against the
  constants themselves (`MINIMAX_MODEL`, `GLM_MODEL`, `CODEX_RUST_MODEL`, `PI_CONTROL_MODEL`) and
  check prices only for being positive, so changing a model id's *value* leaves both suites
  green. A placeholder constant would **not** be rejected. Leaf γ adds the literal pin that makes
  the slate's values enforceable.
- `EventType` has no registry and no full-set-pinning test, so adding a member is free — and
  unbacked by any collision check. A new closed vocabulary must carry its own uniqueness
  assertion. **Verified.**
- `orchestrator/cli.py::main` is a **click** group whose every existing `eval*` command is a flat
  sibling; there is no nested sub-group anywhere, and nothing pins the command set. An
  `eval-shadow` group is a new pattern that breaks nothing. **Verified.**
- The harness's `_maybe_auto_eval` sibling-task redo hook has fired **0 times ever**
  (`plans/author-declared-complexity-prd.md` §Out of scope) and files real tasks; it is
  **not** reused (decision 3).
- Isolation is **not** in place today: eval cells write into live fused-memory,
  escalations and the task DB because `run_architect_eval` leaves
  `strict_mcp_config=False` (task **4757**, pending, high) and eval-lane escalations reach
  the L2 queue (task **3096**, pending). This PRD's dispatching leaves **depend on both**.

## Sketch of approach

A `ShadowCoordinator` lives in each orchestrator process. It watches the event stream the
harness already emits, decides — deterministically, per sampled task and candidate — which
**cells** to open, runs each cell through the existing eval runners against a **live
fixture** built from the production task, and **settles** each cell when production's own
outcome for that task becomes known. Cells are rows in a `shadow_cells` table; results are
`EvalResult` JSONs; the cell's throwaway worktree is **detached and carries no ref at all**,
so a cell has no branch; nothing about a cell is a task, an escalation, a merge request or a
memory write.

### The four shapes

| Shape | Trigger | Candidate runs… | Shared with production | Primary outcome (paired vs production) |
|---|---|---|---|---|
| `implementer` | `phase_exit(PLAN)` on the production task | the implementer, on production's plan.json, at `branch_base_sha` | plan, base | `tests_pass`, `review_blocking_issues`, `iterations`, `cost_usd` to terminal |
| `architect` | `phase_enter(PLAN)` | the architect, live, plan-only | base, briefing | `terminal_kind` split, `plan_quality` (judge, secondary) |
| `architect-consequence` | as `architect`, then on the candidate plan | leg 1: candidate architect; leg 2: the **incumbent** implementer on the candidate's plan | base, implementer | leg-2 `tests_pass`, `review_blocking_issues`, `iterations`, `cost_usd`; judge secondary |
| `end-to-end` | `phase_enter(PLAN)` | candidate architect **and** candidate implementer | base | as `implementer`, plus $/done |

`architect-consequence` is the instrument for Leo's hypothesis: same task, same base, same
implementer; only the plan differs.

**Harness variant axis (decision 14).** A candidate name may carry a harness variant,
`<config>@<variant>` with `variant ∈ {whole-plan, one-step}` (`whole-plan` is today's
brief and the default). Under `one-step` the implementer brief instructs the agent to
complete **exactly the next pending step**, commit, call `mark_step_done`, and stop; the
EXECUTE loop's iteration cap and the per-invocation turn/budget ceilings scale to the plan.
Listing `glm-5.3-endpoint`, `glm-5.3-endpoint@one-step` and `sonnet@one-step` in
`shadow_eval.candidates` yields, per sampled task, a within-task set of production plus
three cells, so the report reads model × variant as paired deltas. `architect` alone stays cheap and answers the decline
question. `end-to-end` is the live confirm stage, opened only for a survivor candidate.

### Life of a cell

`sampled → opened → running → awaiting_reference → settled | cap_excluded | expired | failed`

1. **Sample.** On the trigger event, `should_open(task, candidate, shape)` hashes
   `(task_id, candidate, shape, seed)` against `sample_rate` within the task's stratum
   (`task_sampler.py::classify_kind / classify_path / repo_of_project`), then checks the
   daily USD cap, `max_concurrent`, the host `saturated()` gate and the candidate's
   own key presence. Every negative decision is a `shadow_cell_skipped` event with a
   reason (INV-11).
2. **Open.** Write the `shadow_cells` row, build the live fixture, create the detached eval
   worktree at `branch_base_sha` via `snapshots.py::create_eval_worktree`, passing
   `task_id=f"shadow_{task}_{cell_id}"` — already the fixture `id` of contract C3, so the
   helper is reused byte-unchanged and the cell's path is recorded in `worktree_path`.
3. **Run.** Dispatch through the existing runner for the shape, under the shadow
   invocation profile: `EVAL_PROFILE` + `strict_mcp_config=True` + null-routed memory +
   the landlock sandbox for implementer legs. A usage-cap hit ends the cell as
   `cap_excluded`; there is no retry — production owns the account pool.
4. **Await reference.** On `merge_finalized(state=done)` for `task/<task_id>`, the
   reference is `merge_sha` against `branch_base_sha`. On any other terminal state
   (blocked, cancelled, superseded, expired) the cell settles with `reference_kind=none`.
5. **Settle.** Score per shape (contract C4), write `settled_at`, save the `EvalResult`,
   emit `shadow_cell_settled`. A cell whose production task has no terminal event by
   `settle_deadline` settles as `expired` — never left open (INV-7).

## Resolved design decisions

1. **(Leo) Live paired shadow replaces the frozen corpus as the capability instrument.**
   `evals/tasks/` is retained only as the instrument regression corpus for the runner's own
   tests; `evals/tasks_hard_v2/` is retired as capability evidence. Eval-revival decision 1
   ("fixed target") is amended by the companion task in Phase 4.
2. **(Leo) Four shapes in scope:** `implementer`, `architect`, `architect-consequence`,
   `end-to-end`. `end-to-end` opens only when `shadow_eval.end_to_end_candidates` names
   the candidate — it is the confirm stage, not a screen.
3. **(Leo) Cells are not tasks.** They live in `shadow_cells` (runs.db, owned by
   `run_store.py`) plus `EvalResult` JSONs; they carry no task id of their own, never
   enter the curator, the merge queue, the orphan reapers or the dashboards' task views.
   The `auto_eval` sibling-task path is not reused.
4. **Reference and settle rule.** The reference is production's landed diff. If
   production does not land, the cell is scored on what does not need a reference
   (verify gates, review outcome, plan structure, `terminal_kind`) and tagged
   `reference_kind=none`; it is never scored as a pass or a fail by default.
5. **Pairing is within task.** Every report statistic is a paired difference
   (candidate cell minus production's own metrics for the same task, read through the two
   accessors leaf α builds — `RunStore.get_task_outcome` over `task_results` for the outcome
   and `CostStore.task_invocation_cost` over the `invocations` ledger for cost **[S1]**, never
   from logs and never from SQL hand-written elsewhere), with a bootstrap CI over pairs. Unpaired
   cells (production unreferenced *and* the shape needs a reference) are counted, not
   averaged in.
6. **Cap policy.** A shadow cell that hits a usage cap is `cap_excluded`, not retried and
   not re-sampled. Production always wins the pool (consistent with the 2026-08-30 ruling
   on task 4741: evals share the fleet pool and have no reserve).
7. **Load policy.** A cell counts as a dispatch for the PSI/runqueue gate: it is opened
   only when `saturated()` is false and `shadow_eval.max_concurrent` has a free slot
   (default 1). It never pre-empts a production dispatch.
8. **Sampling is deterministic and stratified.** `should_open` is a pure function of
   `(task_id, candidate, shape, stratum, seed, sample_rate)`, so a cut is reproducible
   and both projects' coordinators agree without coordination.
9. **Isolation is a hard prerequisite, owned upstream.** Dispatching leaves depend on
   4757 (strict MCP + no live-store writes) and 3096 (eval-lane escalation containment).
   Until both land, the coordinator refuses to open cells and says so
   (`shadow_cell_skipped reason=isolation_unavailable`).
10. **Storm escape (INV-4).** The coordinator pauses itself and files exactly one
    escalation when: the daily USD cap trips, or `failed`+`expired` cells reach a streak of
    `shadow_eval.failure_streak_pause` (default 5), or the cell/production cost ratio for
    a candidate exceeds `shadow_eval.cost_ratio_ceiling` over the trailing window.
11. **One home (INV-9).** The `shadow_cells` row is the authoritative record of a cell;
    the `EvalResult` JSON carries the row id and the report renders from rows. No cell
    fact is copied into task metadata, escalations or memory.
12. **Candidate slate refresh is folded in (Leo).** One leaf updates the endpoint and
    Codex model ids and prices in `evals/configs.py`. **Settled at decompose** (Leo, after a
    live market check on 2026-09-10) — these are authoritative constants, not placeholders:
    Codex arms `gpt-6-astra` ($10/$50 per 1M in/out; needs Codex CLI ≥ 0.153.0),
    `gpt-5.6-sol` ($4/$20) and `gpt-5.6-terra` ($2/$12); endpoint arms `glm-5.3`
    ($1.40/$4.40), `glm-5.3-flash` ($0.15/$0.50 **list** — the 50% promo expired
    2026-09-09 24:00 UTC+8 and aggregators still quote it) and `MiniMax-M3`
    ($0.30/$1.20 native, base `https://api.minimax.io/v1`). The March-dated
    `codex-gpt54*` / `gemini-*` entries are retired. **Access is a GLM Coding Plan** (Leo, 2026-09-11,
    `https://docs.z.ai/devpack/overview`), which settles the endpoint: a Coding Plan key works
    **only** via `https://api.z.ai/api/coding/paas/v4` (OpenAI protocol) or
    `https://api.z.ai/api/anthropic` (Anthropic protocol) — the general `/api/paas/v4` returns an
    **error** with a subscription key. γ keeps a startup probe, now asserting that endpoint
    answers rather than discovering which family does. A Coding Plan covers GLM-5.3 and
    GLM-5.3-Flash, so both slate entries run on one subscription. It is **credit-metered, not
    token-metered**, so the GLM entries in `CANDIDATE_ENDPOINT_PRICES` are an **imputed** list
    price and must be tagged as such — see decision 16. **[S7]** γ also adds the literal slate
    pin, because the existing bundle tests assert against the constants they protect and would
    stay green on any value. The `eval-ofat` candidate-selector gap is **not** fixed: the
    coordinator is the screen.
13. **No production config change** is applied by anything in this PRD (as every eval
    PRD in this lineage).
14. **(Leo) The one-step implementer harness variant is an eval axis, with a control.**
    Hypothesis: cheap models fail as drop-in whole-plan implementers but may succeed when
    each invocation does exactly one plan step. The variant is a per-role config knob
    (`implementer_brief_variant`) consumed by `build_implementer_prompt`, carried on
    `EvalConfig.harness_variant` and named `<config>@one-step`; under it
    `max_execute_iterations` becomes `steps + prerequisites + slack` and the implementer's
    per-invocation `max_turns`/`budget_usd` scale down (values: open question 7).
    `sonnet@one-step` is always eligible as the control so a cheap-model result is read
    against what the variant does for a capable model. The production consumer beyond this
    PRD is a routing flip on a favourable report, filed by ruling as in decision 13; the
    knob ships default `whole-plan` and changes nothing until then.
15. **(Leo) The agent harness is an eval axis, and GLM must be measured on its own.** GLM-5.3
    shares a base model with GLM-5.2; Z.ai attributes the whole gain to roughly a further month
    of post-training RL on executable agentic environments with stronger verifiers, and **ZCode
    — Z.ai's own agentic development environment — is marketed as the Official Harness for
    GLM-5.3**. Measuring GLM only under the Claude Code harness therefore measures its ability to
    imitate a Claude-Code-shaped agent, not its capability: the same confound class this PRD
    exists to remove, arriving from the other direction. `EvalConfig.backend` already dispatches
    `claude | codex | gemini | pi` in `agents/invoke.py`, so a `zcode` backend is a fifth arm of
    an existing seam. **The arm is probe-gated** (leaf ω): ZCode is documented as a *desktop*
    ADE, so whether it can be driven headlessly with MCP at all is genuinely in doubt, and an
    unbuildable arm is a real answer to land rather than a backend to half-build. **The
    Claude-harness control for the same model is mandatory** — the existing `pi-sonnet-control`
    candidate is exactly this pattern, and an arm without its control measures nothing.
16. **A subscription-metered arm gets two cost columns, and the modelled one is tagged (Leo).**
    A GLM Coding Plan bills **credits** against a weekly and rolling-5h allowance, not tokens, so
    a GLM cell meters no dollars. Writing `0` would make the cheapest-looking arm appear free and
    corrupt C5's headline statistic; writing a list price into the same field would present a
    price Leo does not pay as a measured cost — the mistake this lineage already corrected once
    when `eval-framework-revival-prd.md` decision 1 retired the `hardware_time_seconds`
    GPU-imputation machinery. So the row carries `cost_credits` **and** `cost_usd`, plus a
    `cost_basis` discriminator (`metered` | `subscription`) that is a typed field, never inferred
    from a candidate name (heuristic 12). The report prints both, tagging every imputed figure at
    the point of display. **The Coding Plan is eval-exclusive at present** (Leo, 2026-09-11), so
    no quota-contention model is built — but the ceiling is real, so exhaustion settles a cell
    `cap_excluded` with reason `credits_exhausted`, and the `cost_ratio_ceiling` storm escape
    compares only within a matching `cost_basis`: across bases it would be permanently
    un-trippable for the subscription arm, a guard present and green and disarmed (INV-4).

## Pre-conditions for activating

- Tasks **4757** and **3096** landed (isolation). Phase 1 leaves may proceed; Phase 2's
  dispatching leaf and everything after it depend on both.
- Provider credentials present in the orchestrator's environment for any endpoint bundle to be
  sampled — a **GLM Coding Plan** key for the GLM arms (paired with the coding endpoint, not the
  general one), `MINIMAX_API_KEY`, a Codex CLI ≥ 0.153.0 for `gpt-6-astra`, … A missing
  credential is a per-candidate `shadow_cell_skipped reason=no_credentials`, never a run that
  401s; a Coding Plan whose allowance is spent settles the cell `cap_excluded` with reason
  `credits_exhausted`.
- Nothing else is novel: every substrate item in §Premise exists on main.

## Cross-PRD relationship

| Other PRD / owner | Direction | Mechanism crossing the seam | Owner of the integration |
|---|---|---|---|
| `eval-framework-revival-prd.md` | **consumes** | `run_eval` / `run_architect_eval` / `run_end_to_end`, `EvalResult`/`EvalMetrics`, `save_result`, the judge, `build_eval_orch_config`, `task_sampler` classifiers | revival owns the runners; **this PRD** owns the live fixture builder and any additive fields on `EvalMetrics` (`shadow_cell_id`, `reference_kind`) |
| task **4757**, task **3096** | **consumes** | strict MCP config on eval invocations; eval-lane escalation containment | those tasks; this PRD's ε2 depends on them and adds the deterministic check that a shadow invocation carries them |
| `fable-architect-trial-v2-prd.md` / esc-3637-1 | **produces** | the `architect-consequence` report for `architect-fable-max` vs `architect-opus-max` | this PRD (κ gate); 3637's ruling consumes it |
| `adaptive-model-routing-prd.md` | **produces** | a favourable report as the evidence a flip task cites | that PRD's flip task, filed by Leo's ruling — never by this PRD |
| `load-throttle-harmonisation-prd.md` | **consumes** | `shared.psi::saturated()` / `tripping_metric()` for the open decision | that PRD; this PRD adds no arm |
| `usage-gate-model-scoped-caps-prd.md` | **consumes** | `UsageGate` via `_build_eval_usage_gate`; `cap_hit` | that PRD; the `cap_excluded` disposition is this PRD's |
| `os-sandbox-worktree-containment-prd.md` | **consumes** | landlock wrapping for implementer legs in the cell's eval worktree | that PRD owns the wrapper (`agents/sandbox_dispatch.py::wrap_command` → `agents/landlock.py::build_landlock_command`). **[S6]** No eval call site passes `sandbox_modules` today, so the wrap never engages for evals — **δ owns that plumbing**; it is not inherited |
| the harness `_maybe_auto_eval` hook | **none** | not reused (decision 3) | — |

## Contract (H)

### C1 — `ShadowCell` record (`run_store.py`, table `shadow_cells`)

```
shadow_cells(
  cell_id TEXT PRIMARY KEY,            -- ulid
  parent_cell_id TEXT,                 -- consequence leg 1 -> leg 2. A real edge, never a
                                       -- cell_id string prefix (heuristic 12; INV-1)
  project_id TEXT NOT NULL,
  task_id TEXT NOT NULL,               -- the production task
  shape TEXT NOT NULL,                 -- implementer|architect|architect-consequence|end-to-end
  candidate TEXT NOT NULL,             -- EvalConfig.name
  incumbent TEXT NOT NULL,             -- the pinned incumbent config name for the role
  stratum TEXT NOT NULL,               -- "<repo>×<kind>×<path>"
  base_sha TEXT NOT NULL,              -- task.metadata.branch_base_sha at open
  worktree_path TEXT,                  -- the DETACHED eval worktree. A cell has no branch [S5]
  plan_source TEXT,                    -- 'production' (implementer) | 'candidate' (consequence leg 2) | NULL
  state TEXT NOT NULL,                 -- sampled|opened|running|awaiting_reference|settled|cap_excluded|expired|failed
  reason TEXT,                         -- populated on cap_excluded|expired|failed
  reference_kind TEXT,                 -- landed|none  (NULL until settled)
  reference_sha TEXT,                  -- merge_sha when landed
  result_path TEXT,                    -- EvalResult JSON once saved
  cost_usd REAL NOT NULL DEFAULT 0,    -- imputed for a subscription-metered arm
  cost_credits REAL NOT NULL DEFAULT 0, -- the REAL figure for a Coding Plan arm
  cost_basis TEXT NOT NULL,            -- 'metered' | 'subscription'. A typed discriminator,
                                       -- never inferred from the candidate name (heuristic 12)
  run_id TEXT,                         -- the orchestrator run that opened it; startup
                                       -- reconciliation reads this to find orphans
  opened_at TEXT NOT NULL, running_at TEXT, settled_at TEXT,
  settle_deadline TEXT NOT NULL,       -- opened_at + shadow_eval.settle_deadline
  owner TEXT NOT NULL DEFAULT 'shadow_coordinator'
)
```

Invariants: a `(task_id, shape, candidate)` triple opens at most once per trial index;
`state` transitions are monotone along the lifecycle above; every non-`settled` terminal
state carries a non-empty `reason`; `settle_deadline` is never NULL and `owner` is never NULL
(INV-7); and **no row stays non-terminal across a restart** — the coordinator reconciles any
`opened`/`running` row whose `run_id` is no longer live to `failed`
(`reason=orphaned_by_restart`) at startup, because the fleet redeploys on an ~8h clock and
soft-cancels every in-flight task, so an abandoned claim is the common case, not an edge
(INV-6 `status-matches-liveness`).

### C2 — `ShadowCoordinator` (new module `orchestrator/shadow_eval.py`)

```python
class ShadowCoordinator:
    def __init__(self, config: ShadowEvalConfig, store: RunStore, events: EventStore,
                 runner: ShadowRunner, clock: Callable[[], datetime], seed: int) -> None: ...

    def should_open(self, task: dict, shape: str, candidate: EvalConfig) -> OpenDecision:
        """Pure. Returns OpenDecision(open: bool, reason: str). Reasons are a closed
        vocabulary: sampled_out | no_credentials | isolation_unavailable | budget_exhausted |
        credits_exhausted | concurrency_full | host_saturated | shape_disabled | already_open."""

    async def on_phase(self, task_id: str, phase: str, entering: bool) -> list[str]:
        """Harness-side hook: called for every phase_enter/phase_exit. Opens cells whose
        trigger matches; returns cell_ids opened. Never raises into the harness."""

    async def on_merge_finalized(self, payload: dict) -> list[str]:
        """Settles awaiting cells for payload['branch']. Returns cell_ids settled."""

    async def expire(self, now: datetime) -> list[str]:
        """Settles every awaiting cell past settle_deadline as expired. The sweep is
        explicitly bounded and NAMES what it deferred: awaiting cells accumulate for up to
        settle_deadline_hours (168h), so the fan-out is not bounded upstream (INV-8), and a
        silent truncation would be indistinguishable from 'nothing left' (INV-11)."""

    async def reconcile_on_start(self) -> list[str]:
        """Writes `failed` (reason=orphaned_by_restart) for every non-terminal row whose
        run_id is no longer live. Without it, a restart strands rows in `opened`/`running`
        with no exit owner at all (INV-6, INV-7)."""

    def storm_state(self) -> StormState:
        """paused: bool, reason: str | None — read by the harness before each on_phase."""
```

Ordering invariants: `on_phase` for a task is idempotent per `(task, shape, candidate)`;
`on_merge_finalized` and `expire` are the **only** writers of `settled_at`; a cell whose
run fails writes `failed` before any settle can occur. Errors inside a cell never propagate
to the production slot (the coordinator wraps the runner; a wrapped failure is a `failed`
cell with the exception class in `reason`).

### C3 — live fixture builder (`evals/live_fixture.py`)

```python
def build_live_fixture(task: dict, *, base_sha: str, project_root: Path,
                       plan: dict | None, verify_commands: dict[str, str],
                       shape: str, cell_id: str) -> dict
```

Returns a dict accepted unchanged by `runner.load_task`'s consumers: `id =
f"shadow_{task_id}_{cell_id}"` (also the `task_id` handed to `create_eval_worktree`),
`pre_task_commit = base_sha`, `task_definition` from the task record's
title/description/details, `plan` as given (production's for `implementer`, the candidate's
for consequence leg 2, `None` for architect legs), **no `reference`** at build time — plus
the wider optional key surface named in §Premise (`setup_commands`, `modules`, `name`,
`max_execute_iterations`, `max_review_cycles`, `judge_after_each_iteration`,
`max_architect_turns`, `adversarial`), each emitted or deliberately omitted rather than
silently dropped.

Invariants: `build_live_fixture` reads nothing from the live repo beyond its arguments — the
plan dict is passed **in** by the caller, read from
`<worktree_base>/.task-meta/<worktree_name>/plan.json` **[S8]**, and the base is the caller's,
never "HEAD now". A shape that requires a plan, called with none, **raises**: it must never
emit a plan-less fixture that would later score as a candidate decline, which would silently
convert a data-plumbing bug into a capability measurement (INV-11 `no-silent-fail-soft`).

### C4 — settle and scoring rules

| Shape | `reference_kind=landed` | `reference_kind=none` |
|---|---|---|
| `implementer`, `end-to-end`, consequence leg 2 | verify gates at `base_sha`; judge vs `reference_sha` diff; composite as `compute_composite` | verify gates + review outcome only; `composite_score` left `None`, never 0.0 |
| `architect` | `terminal_kind`; `plan_quality` judged vs reference diff | `terminal_kind`; `score_plan_structure` only; `plan_quality=None` |

**[S1]** Production's paired metrics for the same task are read through leaf α's two
accessors — `RunStore.get_task_outcome` over `task_results` for the outcome, and
`CostStore.task_invocation_cost` over `shared.cost_store`'s `invocations` ledger for cost.
Those are different owners in different packages that merely share the `runs.db` file, and
neither accessor exists on main today. Never from logs, never from SQL written at the call
site. A cell settles exactly once.

### C5 — report (`orchestrator eval-shadow report`)

Per `(shape, candidate)`: `pairs`, `unpaired`, `cap_excluded`, `expired`, `failed`; the
shape's primary outcome as paired mean difference with a 95% bootstrap CI; **both cost
columns** — `credits per usable outcome` and `$ per usable outcome`, for candidate and incumbent
— with **every imputed figure tagged at the point of display** (decision 16; a row whose
`cost_basis` is `subscription` has a modelled dollar figure, and if the display cannot carry the
tag it omits the dollar figure rather than printing it bare); `terminal_kind` split; and `n_min`
(contract: a line whose `pairs < n_min` is printed with an `UNDERPOWERED` tag, not hidden).
Machine form: `--json` emits the same rows, `cost_basis` included so a machine consumer can also
tell a measured cost from a modelled one.

### C6 — config (`ShadowEvalConfig`, green-tier hot-reloadable, under `shadow_eval:`)

`enabled` (default **false**), `sample_rate` (0..1, per candidate override map),
`shapes` (enabled set), `candidates` (list of `EvalConfig` names, optionally
`<name>@<variant>`), `end_to_end_candidates`,
`max_concurrent` (1), `daily_budget_usd` (50.0 — the `auto_eval_redo_budget_usd` budget
semantics, but **[S2]** *not* its reload tier: that field is restart-only, so `shadow_eval`
is made green-tier by an explicit `_submodel_leaf_paths('shadow_eval', ShadowEvalConfig)`
entry in `config.py::RELOADABLE_FIELDS`),
`settle_deadline_hours` (168), `failure_streak_pause` (5), `cost_ratio_ceiling` (3.0),
`n_min` (12, provisional — see Open questions), `seed`.

## Boundary-test sketch (H) — the integration-gate signal

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Implementer cell opens on PLAN exit | `enabled`, candidate sampled in, plan.json present, isolation landed | one `shadow_cells` row `opened`, its `worktree_path` set to a **detached** eval worktree at `branch_base_sha`; fixture `plan` byte-equal to plan.json |
| 2 | Sampled-out task opens nothing | hash outside `sample_rate` | no row; one `shadow_cell_skipped(reason=sampled_out)` event |
| 3 | Isolation not landed | 4757 or 3096 absent (probe: invocation kwargs lack `strict_mcp_config=True`) | no cell; `skipped(reason=isolation_unavailable)`; coordinator `storm_state.paused=False` |
| 4 | Production lands | cell `awaiting_reference`; `merge_finalized(state=done, merge_sha=X)` | cell `settled`, `reference_kind=landed`, `reference_sha=X`; result JSON path set; one `shadow_cell_settled` |
| 5 | Production blocks and is cancelled | cell awaiting; `merge_finalized` never fires; task terminal `cancelled` observed | cell `settled`, `reference_kind=none`, `composite_score=None` |
| 6 | Deadline passes | cell awaiting; `now > settle_deadline` | `expire()` → state `expired`, reason `settle_deadline`; never re-opened |
| 7 | Cap hit mid-cell | `UsageGate` reports `cap_hit` for the cell's invocation | state `cap_excluded`; no retry invocation recorded; production task unaffected |
| 8 | Host saturated | `saturated()` true at trigger | `skipped(reason=host_saturated)`; production dispatch proceeds |
| 9 | Consequence leg 2 | leg-1 architect cell produced a plan with `plan_steps>0` | a second cell, `plan_source=candidate`, incumbent implementer config, same `base_sha`; leg 1 and leg 2 rows linked by the `parent_cell_id` **column** — never a `cell_id` string prefix (heuristic 12; INV-1) |
| 10 | Consequence leg 1 declines | leg-1 `terminal_kind` is a decline | **no** leg 2; leg 1 settles with the decline recorded; report counts it in the `terminal_kind` split |
| 11 | Storm escape | five consecutive `failed`/`expired` cells | `storm_state.paused=True`; exactly one escalation filed; subsequent triggers `skipped(reason=paused)` |
| 12 | Report renders from rows only | rows present, result JSON deleted for one cell | report prints the row with `result_missing`, exit non-zero; no log-scrape |
| 13 | A cell can never reach the merge lane **[S5]** | a cell has run; its worktree is under `<root>-eval-worktrees/`, a sibling of `project_root` | two limbs, both **executed**, not read from prose: (i) **no ref exists** — `create_eval_worktree` is `git worktree add --detach`, so there is nothing to enqueue and no `merge_request` is ever filed; (ii) the coordinator module imports no merge client. No reaper exclusion is added: all four reapers already filter on `wt.parent == worktree_base` (`.worktrees`), which the eval root is not under |
| 14 | Runner failure is contained | the shape runner raises | cell `failed` with exception class in `reason`; the production slot's `TaskReport` is unchanged |
| 15 | One-step variant is honoured | candidate `X@one-step`, a plan with N pending steps | the cell's config carries `implementer_brief_variant=one-step`; every implementer invocation's brief names exactly one step; on a clean run `iterations == N`, and each iteration-log entry's `steps_completed` **list has length exactly 1** — the field is a list computed as a set difference, so this is the property ν *establishes*, not one today's mechanism has **[S3]**; the iteration cap equals `N + prerequisites + slack`, computed against the same `- progress_resume_total` adjustment the existing cap check uses |

## Decomposition plan

**Filed 2026-09-11 as tasks 5382–5395**, all `pending`, 21 dependency edges. Leaf sizing
follows the overlay bands (300–1,500 LOC, ≤10–12 files). Greek labels below carry their real
ids. **G7** notes are inline. Every leaf cites `docs/code-quality.md` heuristics in its brief
where they bind (small function scopes and stateless interactions for the coordinator; SPOT
for the reason vocabulary). Per-leaf capability→evidence bindings:
`plans/live-shadow-eval-prd.capability-manifest.md` and its YAML sidecar.

**Phase 1 — foundation**

- **α — cell store, config, events and the production-metric accessors** *(leaf, task
  **5382**)*. `shadow_cells` table and `ShadowCell` dataclass in `run_store.py` (extend the
  `_SCHEMA` constant — `CREATE TABLE IF NOT EXISTS` is idempotent, so a new *table* needs no
  `_migrate_*` helper), including the `parent_cell_id` and `worktree_path` columns and `run_id`;
  `ShadowEvalConfig` in `config.py` registered green-tier via
  `_submodel_leaf_paths('shadow_eval', ShadowEvalConfig)` **[S2]**;
  `EventType.shadow_cell_opened/skipped/settled` with the closed reason vocabulary as an enum
  (INV-1: the vocabulary is a schema, not prose); the additive `EvalMetrics.shadow_cell_id` /
  `reference_kind` fields; and **[S1]** the two accessors production's paired metrics are read
  through — `RunStore.get_task_outcome` over `task_results` and
  `CostStore.task_invocation_cost` over `shared.cost_store`'s `invocations`, the latter also
  retiring the hand-written SQL in `harness.py::Harness._auto_eval_budget_used_24h` (SPOT).
  **Signal:** `orchestrator check-config` accepts a `shadow_eval:` block with no unknown-key
  finding (it walks raw YAML through `config.py::census_config_keys`, and `OrchestratorConfig`
  is `extra='ignore'`, so an unregistered block exits 1 — the signal is real, not tautological);
  a hermetic test inserts a row through `RunStore` and reads it back including `parent_cell_id`
  and `worktree_path`; `reload_config` reports `shadow_eval.*` under `applied`; both accessors
  return production's recorded outcome and invocation cost for a seeded task. Modules:
  `orchestrator`, `shared`. Prereqs: none.
- **β — live fixture builder** *(leaf, task **5383**)*.
  `evals/live_fixture.py::build_live_fixture` reusing `task_sampler`'s
  `default_verify_commands` and stratum classifiers. **Signal:** for a synthetic task record +
  plan dict, the emitted fixture round-trips through `runner.load_task`'s consumers and
  `build_eval_orch_config` without error; the builder is proven argument-pure by a test running
  it against a temp dir with no git repo; and a plan-requiring shape called with `plan=None`
  **raises** rather than emitting a plan-less fixture (INV-11). Modules: `orchestrator/evals`.
  Prereqs: none.
- **γ — candidate slate refresh** *(leaf, independent, task **5384**)*. Update
  `evals/configs.py` model ids, base URLs, `CANDIDATE_ENDPOINT_PRICES` and `CODEX_RUST_MODEL`
  to the slate settled in decision 12, retire the 2026-03-19 `codex-gpt54*` / `gemini-*`
  entries from `EVAL_CONFIGS`, and author both Z.ai endpoint constants plus a startup probe
  that fails loudly. **[S7] Signal:** a **new literal slate pin** asserts each candidate's exact
  model string and its `input_per_1m`/`output_per_1m` and goes RED on a stale or placeholder id
  — the existing bundle tests assert against the constants they protect and cannot do this;
  `get_config_by_name` resolves every new name; `claude_endpoint_price_table()` has a priced
  entry for every non-incumbent model. G7: `guards-exercise-behaviour` — the pin is the
  mechanism, not prose about it. Modules: `orchestrator/evals`, `orchestrator/tests`.
  Prereqs: none.
- **ν — one-step implementer brief variant** *(leaf, task **5385**)*.
  `implementer_brief_variant` per-role config knob (green tier; default `whole-plan`), the
  `one-step` branch in `briefing.py::build_implementer_prompt`, bound scaling in
  `_execute_iterations` (subtracting `progress_resume_total`, as the existing cap check does)
  and the per-invocation implementer `max_turns`/`budget_usd` under the variant,
  `EvalConfig.harness_variant` plus the `<name>@<variant>` resolver in `get_config_by_name`
  (which must split the suffix *before* the linear scan and **refuse** an unknown variant rather
  than fall through to `None` — INV-11), and `build_eval_orch_config` mapping the field onto the
  knob. **Signal:** boundary row 15 executed against a real `TaskWorkflow` in eval mode on a
  synthetic 3-step plan; with the knob at its default the rendered brief and every bound are
  byte-identical to today's (the parity tripwire). G7: `contracts-machine-checked` — the variant
  is a validated enum, not prose in the brief. Modules: `orchestrator`, `orchestrator/agents`,
  `orchestrator/evals`. Prereqs: α, γ (both edit files ν also edits — real edges, not advisory).
- **δ — shadow invocation profile** *(leaf, task **5386**)*. `build_shadow_orch_config` =
  `build_eval_orch_config` + `strict_mcp_config=True` + null memory endpoint; **[S6]** the
  `sandbox_modules` plumbing that makes landlock actually engage for implementer legs (no eval
  call site passes it today); plus the **deterministic isolation probe** the coordinator consults
  (row 3): a check that the invocation kwargs a shadow cell would send carry strict MCP and that
  3096's eval-lane escalation containment is active. The probe must inspect the built invocation,
  never a task status — 4757 and 3096 are both still pending (INV-3). **Signal:** a test builds
  the profile and asserts the kwargs carry strict MCP, the null memory endpoint and
  `sandbox_modules`; with 4757/3096 absent the probe returns `isolation_unavailable`. G7:
  `guards-exercise-behaviour` — the probe inspects the built invocation, not a docstring.
  Modules: `orchestrator/evals`, `orchestrator/agents`. Prereqs: α, ν. Depends
  (out-of-batch): **4757**, **3096**.

- **ω — ZCode harness arm** *(leaf, probe-gated, task **5399**)*. **Stage 1 is a gate:** probe
  whether ZCode can be driven by the eval runner at all — a headless/non-interactive invocation,
  MCP server config (the implementer leg calls `mark_step_done`), an argument targeting the
  cell's detached eval worktree, and a per-invocation turn/cost bound. ZCode is documented as a
  *desktop* ADE, so the first two are genuinely in doubt. **If either is absent, stop**: land the
  finding as a committed note under `plans/` and escalate for a ruling on an alternative harness
  (the Coding Plan also supports OpenCode and Cline, both CLI-shaped). An unbuildable arm is a
  real answer; what must not happen is a reader unable to tell "measured impossible" from "not
  done" (INV-11). **Stage 2, conditional:** `_invoke_zcode` as a fifth arm of
  `agents/invoke.py`'s existing backend dispatch, plus a `glm-5.3` candidate with
  `backend='zcode'` against the Coding Plan endpoint — **with the same model on the Claude
  harness as its mandatory paired control** (the `pi-sonnet-control` pattern inverted). **Signal:**
  whichever branch fires, executed — the committed probe finding plus escalation, or a cell for
  the ZCode candidate running through `run_eval` against a real fixture alongside its
  Claude-harness control over the same fixture. Modules: `orchestrator/agents`,
  `orchestrator/evals`, `plans/`. Prereqs: γ (slate constants + Coding Plan endpoint), δ (same
  file: `agents/invoke.py`).

**Phase 2 — vertical slice (the integration gate)**

- **ε1 — coordinator core** *(leaf, task **5387**)*. `orchestrator/shadow_eval.py`:
  `should_open` (pure), the cell state machine, `expire` (bounded, naming what it deferred),
  **`reconcile_on_start`**, storm state, settle rules (C4) as pure functions over `EvalMetrics`
  and a reference. Hermetic tests cover rows 2, 5, 6, 10, 11 with an injected clock and a fake
  runner. G7: `holds-owned-and-bounded` — every awaiting cell has `owner` and `settle_deadline`;
  `status-matches-liveness` — no row stays non-terminal across a restart;
  `no-silent-fail-soft` — every non-open is a reason; `loop-thread-occupancy-bounded` — the
  expiry sweep is capped. **Signal:** the state-machine tests; a property test that `should_open`
  is deterministic in its inputs; and a restart-orphan test showing an `opened`/`running` row
  from a dead run reconciles to `failed` with `reason=orphaned_by_restart`. Modules:
  `orchestrator`. Prereqs: α.
- **ε2 — harness wiring + implementer shape, end to end** *(integration gate, task **5388**)*.
  Hook `on_phase` into `workflow.py::TaskWorkflow._enter_phase`, the sole emitter of both phase
  events (the only pre-existing seam is `_await_cancellable`'s `on_soft_cancel`; this adds the
  first observer). **[S4]** Hook `on_merge_finalized` as a further
  `req.result.add_done_callback` in `merge_queue.py::enqueue_merge_request`, where two already
  coexist — the event itself has no in-process consumer outside the merge worker. Create the
  detached eval worktree via `snapshots.py::create_eval_worktree`, run the `implementer` shape
  through `run_eval`, apply the PSI gate and `max_concurrent` checks, and set `cap_excluded` on
  a `detect_cap_hit` boolean. **[S5]** Row 13 is re-bound: no reaper exclusion is added
  (structurally unnecessary), and the "never reaches the lane" claim is proved by the absence of
  any ref plus the coordinator's lack of a merge client — G6 branch 4, executed rather than
  asserted. **Signal: boundary rows 1, 3, 4, 7, 8, 13, 14, 15 executed against a real
  `TaskWorkflow` in eval mode** (the revival's `build_workflow` factory), with the production
  slot's `TaskReport` asserted unchanged. G7: `status-matches-liveness` — a cell whose runner
  dies writes `failed` through the coordinator before the slot returns;
  `loop-thread-occupancy-bounded` — the runner is awaited as a separate task, never inline in
  the slot. Modules: `orchestrator`, `orchestrator/evals`. Prereqs: β, δ, ε1, ν (row 15 runs as
  a shadow cell here).
- **θ1 — minimal report** *(leaf, task **5389**)*. `eval-shadow report` over the `implementer`
  shape: pairs, paired mean difference with bootstrap CI, `$ per usable`, counts,
  `UNDERPOWERED` tag, `--json`. Row 12. The CLI is **click** and every existing `eval*` command
  is a flat sibling, so `eval-shadow` is a new sub-group; nothing pins the command set.
  **Signal:** against a seeded `shadow_cells` fixture the report prints the contracted columns
  and exits non-zero on a missing result JSON. Modules: `orchestrator/evals`,
  `orchestrator/cli`. Prereqs: α (rows), ε1 (settle rules).

**Phase 3 — the remaining shapes**

- **ζ — architect and architect-consequence shapes** *(leaf, task **5390**)*. Trigger on
  `phase_enter(PLAN)`; leg 1 through `run_architect_eval` with `terminal_kind`; leg 2 opened
  only on `plan_steps>0`, through `run_eval` with `plan_source=candidate` and the incumbent
  implementer, **linked to leg 1 by the `parent_cell_id` column**; rows 9 and 10. **Signal:**
  rows 9–10 executed end to end; the report shows the consequence shape with leg-2 outcomes
  paired against production's implementer metrics for the same task. G7:
  `contracts-machine-checked` — the leg link is a column, not a string convention. Modules:
  `orchestrator`, `orchestrator/evals`. Prereqs: ε2, θ1.
- **η — end-to-end shape** *(leaf, task **5391**)*. Through `run_end_to_end`, which already
  takes the two role configs separately, gated on `end_to_end_candidates`. **Signal:** a cell
  for a listed candidate runs both roles live and settles with `$/done` paired; an unlisted
  candidate is `skipped(reason=shape_disabled)`. Prereqs: ζ.
- **θ2 — full report and the η-gate view** *(leaf, task **5392**)*. All shapes;
  `terminal_kind` split; per-stratum breakdown; observed pair variance and iterations-per-step;
  the `architect-consequence` view for `architect-fable-max` vs `architect-opus-max` that
  esc-3637-1 consumes. **Signal:** the report over a seeded multi-shape store matches a
  committed golden output rendered from that store (INV-10 tier 1: it executes the renderer);
  `--json` validates against a committed schema. Prereqs: ζ, η.

**Phase 4 — operator gates and companion corrections**

The PRD originally carried a single κ. It is **split**: one task would hold an open L2 for the
whole campaign — a hold whose exit owner cannot yet act (INV-7) — and a deterministic
`before_done.kind='predicate'` cannot express "escalate when the data arrives", because
predicate mode is check-then-**done**-or-escalate, so a passing check closes the task silently
and resolving the escalation re-runs the check and re-escalates.

- **κ1 — campaign activation gate** *(deterministic pure gate; born-at-L2 for Leo; task
  **5393**)*. Leo rules the `sample_rate`, the candidate slate and the enabled shapes, then sets
  `shadow_eval` in both `/home/leo/src/dark-factory/dark-factory-orchestrator.yaml` and
  `/home/leo/src/reify/dark-factory-orchestrator.yaml` and hot-applies each with
  `reload_config`, reading the returned `applied` / `restart_required` dispositions rather than
  the top-level `reloaded` flag. **Signal:** the escalation is filed and resolved with
  `shadow_eval.*` confirmed `applied` on both running orchestrators. No production config change
  beyond enabling the measurement. Prereqs: θ2, ω (the slate Leo rules on must
  include, or knowingly exclude, the ZCode arm), and 4757/3096 landed (δ's probe is what tells
  the truth about that, not the task statuses).
- **κ2 — the η-gate ruling** *(deterministic pure gate with a `delayed` milestone of 14 days
  anchored on κ1 reaching `done`; task **5394**)*. The gate criterion is `pairs ≥ n_min` for the
  fable consequence shape — and that threshold has **one home**, the report's own `UNDERPOWERED`
  tag (C5); κ2 points at it rather than re-implementing the comparison (INV-9). Read the observed
  pair variance θ2 prints alongside it and state in the ruling what power the campaign actually
  had; if the view is still UNDERPOWERED at 14 days, extending the campaign is a legitimate
  ruling. **Signal:** the escalation filed with the report attached; Leo rules esc-3637-1 on it.
  No config flip. Prereqs: κ1.
- **λ — companion corrections** *(leaf, docs, `complexity='simple'`; task **5395**)*. Amend
  `eval-framework-revival-prd.md` decision 1 and status; append a dated pointer note under
  `fable-architect-trial-v2-prd.md`'s TERMINATED status blockquote, following that file's
  existing convention; add an `OPERATIONS.md` §"Shadow eval" — as **`## 7a.`**, following the
  existing `6a.` precedent, so no heading below it is renumbered; and add the
  retired-for-capability marker as a row in
  `orchestrator/src/orchestrator/evals/tasks_hard_v2/README.md`'s existing table (`evals/tasks/`
  has no README, and a one-line marker does not justify creating one). **Signal:** the four
  documents carry the dated pointers; no restated contract (INV-9). Prereqs: ε2.

## Out of scope

- **Any production routing change.** A favourable report is evidence for a flip task
  under `adaptive-model-routing-prd.md`, filed by Leo's ruling.
- **Fixing `eval-ofat`'s candidate selector** (decision 12) and any further offline
  campaign; the offline runners are kept as the instrument this PRD drives, not as a
  screen.
- **The replay-frame work, task 4844** (4758 is `deferred`, coalesced into 4844 along with
  4765; 4844 is the live successor). It concerns the offline instrument; this PRD neither
  depends on nor closes it (Leo may re-disposition it once κ1/κ2 run).
- **vLLM / self-hosted serving**, the LME programme, the memory-eval programme.
- **Dashboard panels** for shadow cells — a follow-up once θ2's `--json` exists.
- **A shadow reviewer or judge shape.** The judge OFAT axis stays offline.

## Open questions

Resolved at decompose on 2026-09-11 are struck through with their resolution; the rest stand.

1. ~~**Exact provider model strings and list prices for γ.**~~ **RESOLVED** (Leo, after a live
   market check on 2026-09-10) — see decision 12 for the settled slate. γ is authored with
   authoritative constants, not placeholders. **Endpoint also RESOLVED** (Leo, 2026-09-11): access is a
   **GLM Coding Plan**, so the constant is `https://api.z.ai/api/coding/paas/v4` (OpenAI protocol)
   or `https://api.z.ai/api/anthropic` (Anthropic protocol) — the general `/api/paas/v4` errors
   with a subscription key. γ keeps the startup probe as a cheap fail-loud assertion. The
   consequence — credit metering rather than token metering — is decision 16, not an open
   question.
2. **`n_min` calibration.** 12 pairs is a screen floor chosen for legibility, not a power
   calculation — **confirmed unchanged by Leo at decompose**. **Suggested resolution:** θ2
   prints the observed pair variance so κ2's ruling can state the power it had; recalibrate in a
   follow-up.
3. ~~**Where the coordinator reads production's paired metrics.**~~ **RESOLVED** — and the
   substrate was mis-stated when the question was written **[S1]**. The invocation ledger is
   `shared/src/shared/cost_store.py::CostStore`'s `invocations` table, not part of
   `run_store.py`, and neither store has a per-task outcome accessor today. Leaf α builds
   `RunStore.get_task_outcome` (outcome) and `CostStore.task_invocation_cost` (cost); ε1 and θ1
   read only through those.
4. **Settle-deadline default.** 168 h assumes production lands within a week; long
   `merge-deferred` holds may exceed it. **Suggested resolution:** keep 168 h, count
   `expired` in the report, revisit after κ2.
5. **Cross-project report aggregation.** Each orchestrator writes its own runs.db;
   `eval-shadow report --project-root` per project vs a merged view. Decide in θ1.
6. **Consequence leg 2 reviewer.** Whether leg 2 runs the reviewer (adds cost, gives
   `review_blocking_issues`) or stops at verify. **Suggested resolution:** run it —
   review outcome is half the signal for "better plan". Decide in ζ.
7. **One-step bound values.** The per-invocation `max_turns`/`budget_usd` under
   `one-step` and the iteration `slack`. **Suggested resolution:** turns and budget at
   one third of the whole-plan role defaults, slack of 3; ν records the chosen values
   with their basis and θ2 reports iterations-per-step so κ2 can recalibrate.
