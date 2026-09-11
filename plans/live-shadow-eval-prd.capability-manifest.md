# Capability manifest — `plans/live-shadow-eval-prd.md`

Built at decompose, 2026-09-10/11, against main `fcd379ba4f` (PRD) with substrate
re-measured on main the same session. Mechanizes G3 + G6 per leaf: every capability a
leaf's user-observable signal asserts is bound to evidence. A FAIL binding blocks the
batch; an OPEN binding is a decision this leaf owns as its own work product.

Machine-readable twin: `plans/live-shadow-eval-prd.capability-manifest.yaml` (path strictly
derived from the PRD path; `commit_planning` stamps its `task_id` fields and copies the
mechanical `delivered_check`s into producer `metadata.delivered_checks`).

Code cited as `path::symbol` throughout — never `file:line` (CLAUDE.md).

## Substrate corrections made at decompose

Eight PRD §Premise / §Boundary claims were re-measured and did not hold as written. Each
is resolved below and carried into the owning leaf's brief. The PRD itself is NOT edited
(its §Premise is a dated snapshot); this manifest is the correction's home.

| # | PRD claim | Measured | Resolution |
|---|---|---|---|
| S1 | production paired metrics read from "`task_results` / the invocation ledger (`runs.db`)" | the invocation ledger is `shared/src/shared/cost_store.py::CostStore` (table `invocations`), a separate class in the `shared` package that merely shares the `runs.db` file with `RunStore`. Neither store has a per-task **outcome** accessor, and `RunStore.get_task_cost` returns an aggregate cost float only | **α** owns two new accessors (per-task outcome over `task_results`; per-task invocation cost over `invocations`). ε1/θ1 consume them rather than hand-writing SQL (heuristic 7 *stateless interactions*, heuristic 9 *coherent narrow interfaces*) |
| S2 | `ShadowEvalConfig` is green-tier "the `auto_eval_redo_budget_usd` pattern" | `orchestrator/src/orchestrator/config.py::OrchestratorConfig.auto_eval_redo_budget_usd` is **not** in `RELOADABLE_FIELDS` — it is restart-tier. Copying its pattern yields a restart-only knob | **α** registers the submodel explicitly via `_submodel_leaf_paths('shadow_eval', ShadowEvalConfig)` in `config.py::RELOADABLE_FIELDS`; the signal asserts `reload_config` reports `shadow_eval.*` under `applied` |
| S3 | boundary row 15: "each iteration's `iteration_log` entry lists exactly one `steps_completed`" | `steps_completed` is a **list** computed as a set difference of before/after `done` step ids in `workflow.py::TaskWorkflow._execute_iterations` — zero, one or many per iteration. The claim is not a property of today's mechanism | restated as the behaviour **ν delivers**: under `one-step`, every iteration entry's `steps_completed` list has `len == 1`. ν's brief also records that the iteration cap check subtracts `metrics.progress_resume_total`, so the scaled bound must too |
| S4 | "hook `on_merge_finalized` into the merge-lane landing path" | `merge_finalized` has **no consumer inside the orchestrator process outside the merge worker**. `TaskWorkflow` takes its merge outcome from the awaited `MergeRequest.result` future, not the event | **ε2** attaches at the existing plural-callback seam: a further `req.result.add_done_callback(...)` in `orchestrator/src/orchestrator/merge_queue.py::enqueue_merge_request`, beside the retention `_on_finalized` closure |
| S5 | branches are `shadow/<task>/<cell_id>`; row 13 asserts such a branch never reaches the lane | `evals/snapshots.py::create_eval_worktree` runs `git worktree add --detach` — it creates **no ref at all**. There is also no branch-prefix allowlist anywhere in the merge path, so no rejection mechanism exists to bind a "branch is refused" claim to | **the branch is deleted from the design.** A cell reuses `create_eval_worktree` unchanged, passing `task_id=f'shadow_{task}_{cell_id}'` (which is already C3's fixture `id`), and `shadow_cells` records `worktree_path`, not a branch. Row 13's rejection claim becomes structural and executable: no ref is created, and the coordinator imports no merge client |
| S6 | cross-PRD table: landlock wrapping for implementer legs — "this PRD passes the worktree path, nothing more" | eval invocations never pass `sandbox_modules` to `orchestrator/src/orchestrator/agents/invoke.py`, and the whole wrap path in `_invoke_claude_with_sandbox` is gated behind `if sandbox_modules is not None:`. `evals/profile.py::EVAL_PROFILE` carries no sandbox key. **Sandboxing does not engage for any eval invocation today** | **δ** owns the plumbing: the shadow invocation profile passes `sandbox_modules` through to `agents/sandbox_dispatch.py::wrap_command`. Not free, not pre-existing |
| S7 | γ's signal: "`test_eval_candidate_bundles.py` and `test_eval_codex_pi_bundles.py` green against the new constants" | both suites assert against the **constants themselves** (`MINIMAX_MODEL`, `GLM_MODEL`, `CODEX_RUST_MODEL`…), never against literal model ids, and check prices only for `> 0`. Changing a constant's *value* leaves them green — the signal is **vacuous** (INV-10 `guards-exercise-behaviour`: the guard reads the thing it is meant to protect) | **γ** adds a slate pin asserting each candidate's exact `model` string and its `input_per_1m`/`output_per_1m`, following the existing candidate-set pin precedent (tasks 2861/3627). The vacuous clause is dropped from the signal |
| S8 | `.task-meta/<task>/plan.json` | `artifacts.py::TaskArtifacts.write_plan` writes `self.root / 'plan.json'` where the meta root is `TaskArtifacts.meta_root_for(worktree_base, worktree_name)` → `<worktree_base>/.task-meta/<worktree_name>/plan.json` — keyed by **worktree name** | **β**'s brief cites the worktree-name-keyed shape; the builder still receives the plan dict from its caller and reads nothing off the live repo |

Two further out-of-batch facts, recorded but not blocking: task **4758** is `deferred`
(coalesced into **4844**, which is the pending successor), and task **3637** — the
consumer of the `architect-consequence` report — is `blocked` on **3636**, not pending.

---

## α — cell store, config, events and the production-metric accessors

| Capability | Evidence | Verdict |
|---|---|---|
| `RunStore` schema is extensible by a new table | `orchestrator/src/orchestrator/run_store.py::RunStore` builds tables from the module constant `_SCHEMA` via `executescript`; `CREATE TABLE IF NOT EXISTS` is idempotent, so a brand-new table needs no `_migrate_*` method (those exist only for adding columns). `shadow_cells` appears nowhere in the repo — greenfield | PASS |
| green-tier registration exists and is a single call | `orchestrator/src/orchestrator/config.py::RELOADABLE_FIELDS` unions `_submodel_leaf_paths(<name>, <Cls>)` per whole submodel (`routing`, `unblock_auto` registered this way) | PASS |
| `check-config` sees an unregistered block | `orchestrator/src/orchestrator/cli.py::check_config` walks raw YAML through `config.py::census_config_keys`; `OrchestratorConfig` is `extra='ignore'`, so an unregistered `shadow_eval:` block is reported as unknown (exit 1). The signal is therefore real, not tautological | PASS |
| a new `EventType` member needs no registry edit | `orchestrator/src/orchestrator/event_store.py::EventType` is a `StrEnum`; the dashboard deliberately uses string literals rather than importing members, and no full-set-pinning test exists (`test_all_event_types_valid` asserts only `value == name`). Addition is free — and unbacked by any collision check, which α's own reason-vocabulary enum must therefore carry | PASS |
| per-task production **outcome** is readable | `task_results` is written by `RunStore.save_task_result`/`save_run`; its only accessor is `RunStore.get_task_cost`, which returns cost. **No outcome accessor exists** — α builds one | OPEN (α owns it) |
| per-task production **invocation cost** is readable | `shared/src/shared/cost_store.py::CostStore` owns table `invocations` (`task_id`, `cost_usd`, `role`, `model`, `capped`, …) but exposes only window aggregates; `harness.py::Harness._auto_eval_budget_used_24h` hand-writes SQL against `CostStore._require_conn()`. **No per-task accessor exists** — α builds one, and doing so retires one hand-written-SQL site (heuristic 11 SPOT) | OPEN (α owns it) |
| `EvalMetrics` tolerates additive fields | `orchestrator/src/orchestrator/evals/metrics.py::EvalMetrics` is a plain `@dataclass`, every field defaulted, `to_dict` is a bare `asdict`; every read-back site whitelists via `__dataclass_fields__` (`evals/rereview.py`, `evals/runner.py::load_results`). No schema, no full-set test | PASS |

## β — live fixture builder

| Capability | Evidence | Verdict |
|---|---|---|
| `load_task`'s consumers accept a synthesised dict | `orchestrator/src/orchestrator/evals/runner.py::load_task` returns a plain dict; all consumers read by key with `.get` defaults | PASS |
| the full key surface is known | beyond the PRD's eight, consumers also read `setup_commands`, `modules`, `name`, `max_execute_iterations`, `max_review_cycles`, `judge_after_each_iteration`, `max_architect_turns`, `adversarial`. `run_eval` raises `ValueError` when `plan` is falsy | PASS (β's brief carries the full list) |
| verify-command derivation is reusable | `orchestrator/src/orchestrator/evals/task_sampler.py::default_verify_commands(repo)` returns a fresh per-repo copy; `repo_of_project` raises on an unrecognised project | PASS |
| stratum classifiers are reusable | `task_sampler.py::classify_kind`, `::classify_path`, `::repo_of_project` | PASS |
| plan provenance | `orchestrator/src/orchestrator/artifacts.py::TaskArtifacts.write_plan` + `::meta_root_for`; writer call site `orchestrator/src/orchestrator/mcp/plan_tools.py::_create_plan` | PASS (see S8) |
| a missing plan is REFUSED, not silently plan-less | **no such refusal exists in the builder** (it does not exist yet). β owns it: a shape requiring a plan with no plan raises rather than emitting a fixture that would score as a decline (INV-11 `no-silent-fail-soft`) | OPEN (β owns it) |

## γ — candidate slate refresh

| Capability | Evidence | Verdict |
|---|---|---|
| the constants γ rewrites exist | `orchestrator/src/orchestrator/evals/configs.py::CANDIDATE_ENDPOINT_PRICES`, `::CODEX_RUST_MODEL` (currently `'gpt-5.6'`), `::claude_endpoint_candidates`, `::codex_pi_candidates`, `::claude_endpoint_price_table`, `::EVAL_CONFIGS`, `::ARCHITECT_EVAL_CONFIGS` | PASS |
| the entries being retired exist | `EVAL_CONFIGS` carries `codex-gpt54-xhigh` (`gpt-5.4`), `codex-gpt54mini-xhigh` (`gpt-5.4-mini`), `gemini-31-pro-high` (`gemini-3.1-pro-preview`), `gemini-3-flash-high` (`gemini-3-flash-preview`), all added **2026-03-19** (`d3b14de8107`) — March-dated, older than the PRD's "April" | PASS |
| the price table is covered for every candidate | `orchestrator/tests/test_eval_candidate_bundles.py::TestClaudeEndpointPriceTable::test_has_a_valid_entry_for_every_non_incumbent_candidate_model` derives the model set from `claude_endpoint_candidates()` itself | PASS |
| **the slate tests reject a wrong model id** | **rejection-absent.** Both suites assert against the constants, not literals; prices are checked only `> 0`. A stale or placeholder id stays green | FAIL → resolved: γ adds the pin (S7). Recorded as a rejection-mechanism binding γ itself produces |
| the provider strings are authoritative | supplied by Leo at decompose after a live market check (2026-09-10): `gpt-6-astra` $10/$50, `gpt-5.6-sol` $4/$20, `gpt-5.6-terra` $2/$12, `glm-5.3` $1.40/$4.40, `glm-5.3-flash` $0.15/$0.50 (the 50% promo expired 2026-09-09 24:00 UTC+8 — aggregators still quote $0.075/$0.25), `MiniMax-M3` $0.30/$1.20 native | PASS |
| the Z.ai base URL is knowable | **two non-interchangeable endpoint families** — a Coding Plan key needs `https://api.z.ai/api/coding/paas/v4`, a general key `https://api.z.ai/api/paas/v4` (Anthropic-format: `https://api.z.ai/api/anthropic`). Which key the host holds is unknown, and the wrong one 4xxs per cell | OPEN (Leo's ruling at decompose: γ authors both and a startup probe that resolves which the key answers on, failing loudly) |

## ν — one-step implementer brief variant

| Capability | Evidence | Verdict |
|---|---|---|
| the EXECUTE loop is bounded by pending steps | `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._execute_iterations` loops `while reuse_entry_probe or self.artifacts.get_pending_steps()` and caps on `metrics.execute_iterations - metrics.progress_resume_total >= config.max_execute_iterations` | PASS |
| the brief is assembled from plan steps | `orchestrator/src/orchestrator/agents/briefing.py::BriefingAssembler.build_implementer_prompt` partitions `plan['steps']` and `plan['prerequisites']` into done/pending; tail reads *"Execute the next pending steps in TDD order. Commit after each step. Call `mark_step_done(step_id, commit_sha)` to record progress. Stop at a logical boundary."* | PASS |
| a per-role knob can reach the brief | `BriefingAssembler.__init__` stores the whole `OrchestratorConfig` as `self.config`; there is no narrower per-role sub-config. The knob is a new `OrchestratorConfig` field read inside the method, or an explicit parameter | PASS |
| `EvalConfig` tolerates `harness_variant` | `orchestrator/src/orchestrator/evals/configs.py::EvalConfig` is a plain `@dataclass` | PASS |
| `<name>@<variant>` can be resolved | `configs.py::get_config_by_name` is a linear scan over `EVAL_CONFIGS`, `FINAL_RUN_CONFIGS`, `ARCHITECT_EVAL_CONFIGS`, `claude_endpoint_candidates()`, `codex_pi_candidates()`, returning `None` on a miss (never raising) — so the resolver must split the suffix before the scan and must REFUSE an unknown variant rather than fall through to `None` (INV-11) | PASS |
| the eval config maps onto per-role config | `orchestrator/src/orchestrator/evals/runner.py::build_eval_orch_config` returns `apply_eval_profile(base).model_copy(update={...})`, deriving only the implementer role from `EvalConfig` | PASS |
| **"exactly one `steps_completed` per iteration"** | `_execute_iterations` writes `'steps_completed': sorted(completed_after - completed_before)` — a list of 0..n. Not true today | PASS as a **producer** claim: ν is the leaf that makes `len == 1` hold under the variant (S3). Asserted by execution against a real `TaskWorkflow`, not read from the brief text (INV-10) |
| the default path is unchanged | parity tripwire: with the knob at `whole-plan`, the rendered brief and every bound are byte-identical to today. Precedent: the repo's existing candidate/parity pin discipline | PASS |

## δ — shadow invocation profile + isolation probe

| Capability | Evidence | Verdict |
|---|---|---|
| the eval profile exists and null-routes memory | `orchestrator/src/orchestrator/evals/profile.py::EVAL_PROFILE` sets `fused_memory.url` to `orchestrator/src/orchestrator/fm_retry.py::FM_NULL_SENTINEL_URL` (`http://127.0.0.1:1`) | PASS |
| `strict_mcp_config` is the isolation lever | root-caused on task **4757**: `evals/runner.py::run_architect_eval` builds a plan-tools-only `mcp_config` but leaves `strict_mcp_config` False, so the CLI ambient-merges the worktree's `.mcp.json` (live escalation 8102, live fused-memory 8002) | PASS |
| 4757 / 3096 are upstream, not downstream | both **pending** on main; wired as real out-of-batch `add_dependency` edges on δ. Neither is landed — the probe must therefore measure the built invocation, never a task status | PASS (DAG-direction) |
| **landlock wrapping already reaches eval invocations** | **producer-absent.** `evals/runner.py`, `compare.py`, `judge.py` all call `agents/invoke.py::invoke_agent` without `sandbox_modules`; the wrap in `_invoke_claude_with_sandbox` is gated on it being non-None | FAIL → resolved: δ owns the plumbing to `agents/sandbox_dispatch.py::wrap_command` (S6) |
| the probe inspects behaviour, not prose | the probe asserts over the **built invocation kwargs** a shadow cell would send (INV-10 tier 1: execute, don't match text) | PASS |

## ε1 — coordinator core

| Capability | Evidence | Verdict |
|---|---|---|
| the store rows and reason vocabulary exist | producer **α**, upstream | PASS (DAG-direction) |
| production paired metrics are readable | producer **α**'s two accessors, upstream (S1) | PASS (DAG-direction) |
| the host load gate exists | `orchestrator/src/orchestrator/scheduler.py::Scheduler._phase_psi_gate`; `shared/src/shared/psi.py::PsiSample.saturated(cfg)` — a **method on the frozen dataclass**, not a module function, and `False` whenever `read_ok` is False (fail-open) | PASS |
| every non-open carries a reason | closed vocabulary from α as an enum (INV-11, INV-1) | PASS |
| a storm escape exists | streak / daily-cap / cost-ratio pause filing exactly one escalation (INV-4) | PASS (ε1 produces) |
| **a non-terminal row survives a process restart coherently** | **no mechanism.** `expire()` as specified settles only `awaiting_reference` rows past deadline; a row left `opened`/`running` by a restart has no exit owner — and the fleet redeploys on an ~8h clock, soft-cancelling every in-flight task, so this is the common case, not the edge | FAIL → resolved: ε1 adds startup reconciliation writing `failed` (reason `orphaned_by_restart`) for non-terminal rows of a dead run (INV-6 `status-matches-liveness`, INV-7 `holds-owned-and-bounded`) |
| the expiry sweep is bounded | `awaiting_reference` rows accumulate for up to `settle_deadline_hours` (168h default), so the sweep's fan-out is not upstream-bounded; it must cap and name what it dropped (INV-8, INV-11 — no silent truncation) | OPEN (ε1 owns the bound) |

## ε2 — harness wiring + implementer shape (the B+H integration gate)

| Capability | Evidence | Verdict |
|---|---|---|
| a phase hook point exists | `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._enter_phase` is the **sole** emitter of both `phase_exit` and `phase_enter`. Confirmed: `_await_cancellable(..., on_soft_cancel=...)` is the only existing callback-style seam on the harness; `event_store.py` has no subscribe API | PASS |
| a merge-landing hook point exists | `orchestrator/src/orchestrator/merge_queue.py::enqueue_merge_request` already registers **two** independent `req.result.add_done_callback(...)` callbacks (the retention `_on_finalized` emitter and a chain-counter cleanup), so a third is the established pattern (S4) | PASS |
| the landing payload carries what settle needs | read at the **emit site**, not the docstring: `_on_finalized` emits `data={request_id, branch, state, snapshot_tip, merge_sha, superseded_by, generation, reason, landed_via_chain}`. `event_store.py::EventStore.latest_merge_finalized(request_id=, branch=, task_id=)` and `::fetch_events_by_type` both exist | PASS |
| the dispatch base sha is on the task record | `workflow.py::TaskWorkflow._setup_worktree_and_artifacts` writes `{'branch_base_sha': base_commit}` right after worktree creation, soft-failing on error — so a cell must treat its absence as `skipped`, not guess HEAD | PASS |
| the eval worktree is created outside `.worktrees/` | `evals/snapshots.py::create_eval_worktree` → `eval_worktree_root(project_root)` = `<project_root.parent>/<project_root.name>-eval-worktrees`, then `/<task_id>/run-<8hex>`. Verified exactly as the PRD claims | PASS |
| a cap hit is detectable | `shared/src/shared/usage_gate.py::UsageGate.detect_cap_hit(...) -> bool` (the `orchestrator` module of the same name is a re-export shim). It is a **boolean return**, not an exception; `SessionBudgetExhausted` is a different thing and must not be conflated. Built for eval by `evals/runner.py::_build_eval_usage_gate`, which returns `None` fail-open | PASS |
| the production slot object is identifiable | `orchestrator/src/orchestrator/harness.py::TaskReport`, constructed by `Harness._run_slot` | PASS |
| **row 13 — a shadow cell can never reach the merge lane** | rejection-mechanism, rebound after S5. Structural, two limbs, both executable: (i) `create_eval_worktree` runs `git worktree add --detach`, creating **no ref**, so there is nothing to enqueue; (ii) the coordinator imports no merge client. The four worktree reapers (`git_ops.py::prune_stale_merge_worktrees`, `::reap_interactive_worktrees`, `harness.py::_reap_orphan_worktrees`, `::_run_interactive_worktree_reaper_pass`) all filter on `wt.parent == worktree_base` (`GitConfig.worktree_dir`, default `.worktrees`), and the eval root is a **sibling of project_root** — so no exclusion needs adding; the assertion executes rather than asserting prose | PASS |
| a runner failure cannot reach the production slot | ε2 produces: the coordinator wraps the runner, writes `failed`, and the slot's `TaskReport` is asserted unchanged (INV-6; INV-8 — awaited as a separate task, never inline in the slot) | PASS (producer) |

## θ1 — minimal report

| Capability | Evidence | Verdict |
|---|---|---|
| a CLI seam exists | `orchestrator/src/orchestrator/cli.py::main` is a **click** group; every existing command (`::eval_cmd` `@main.command('eval')`, `::eval_ofat_cmd` `@main.command('eval-ofat')`, plus `eval-list-fixtures`/`eval-sample`/`eval-matrix`/`eval-confirm`) is a **flat sibling** — there is no nested group anywhere. `eval-shadow report` is therefore a new sub-group pattern (`@main.group('eval-shadow')`), which θ1's brief must state | PASS |
| no golden pins the command set | no test asserts `main.commands.keys()` or a bare `--help` golden; `test_eval_driver_cli.py::test_eval_help_still_lists_judge_trials_vllm_options` pins only `eval --help` option names. Adding a sub-group breaks nothing — and is caught by nothing | PASS |
| rows, not logs, are the source | `shadow_cells` (α) + the two production-metric accessors (α). INV-2 `structured-facts-at-failure` | PASS (DAG-direction) |
| a missing result JSON is distinguishable | row 12: prints `result_missing` and exits non-zero — a refusal, not a log line (INV-11) | PASS (producer) |
| `n_min` has a stated basis | **bound=12 pairs is a legibility floor, not a power calculation** — PRD open question 2, confirmed unchanged by Leo at decompose. It is not an accuracy bound over a method with an analytical floor, so the floor check does not apply; it is bound as provisional and θ2 prints observed pair variance so κ2's ruling can state the power it had | PASS (provisional, basis stated) |

## ζ — architect and architect-consequence shapes

| Capability | Evidence | Verdict |
|---|---|---|
| the architect runner exists | `evals/runner.py::run_architect_eval(task_path, config, base_config=None, trial=1, timeout_override=None, memory_endpoint=None)`; reads `reference.post_task_commit` for the judge's ground-truth diff | PASS |
| the implementer runner exists for leg 2 | `evals/runner.py::run_eval(...)` — raises `ValueError` if the fixture's `plan` is falsy, which is exactly leg 2's contract (the candidate's plan must be non-empty) | PASS |
| `terminal_kind` is populated | `evals/metrics.py::EvalMetrics.terminal_kind` (task 4760) | PASS |
| the incumbent implementer config is resolvable | `configs.py::get_config_by_name` (linear scan, `None` on miss) | PASS |
| **leg 1 ↔ leg 2 are linked** | the PRD links them "by `cell_id` prefix" — routing on a string prefix, i.e. an ad-hoc parser of an internal value (code-quality heuristic **12 structured data instead of meaningful strings**; INV-1 `contracts-machine-checked`) | FAIL → resolved: **α** carries a real `parent_cell_id` column and ζ sets/reads it. No prefix parsing anywhere |

## η — end-to-end shape

| Capability | Evidence | Verdict |
|---|---|---|
| the end-to-end runner exists | `evals/runner.py::run_end_to_end(task_path, arch_config, impl_config, ...)` — takes the two role configs separately, which is exactly the shape η needs | PASS |
| gating on a named list | `shadow_eval.end_to_end_candidates` from α's config block, upstream | PASS (DAG-direction) |
| an unlisted candidate is refused audibly | `skipped(reason=shape_disabled)` from α's reason enum (INV-11) | PASS (DAG-direction) |

## θ2 — full report and the η-gate view

| Capability | Evidence | Verdict |
|---|---|---|
| all shapes have settled rows | producers ζ and η, upstream | PASS (DAG-direction) |
| the fable candidates exist | `configs.py::ARCHITECT_EVAL_CONFIGS` carries `architect-opus-high`, `architect-opus-max`, `architect-sonnet-high`, `architect-fable-high`, `architect-fable-max` (`claude-fable-5`) — the exact pair esc-3637-1 consumes | PASS |
| the golden is a behaviour check | the golden renders from a **seeded store**, so it executes the renderer rather than matching prose (INV-10 tier 1); the `--json` schema is the contract's one home (INV-1, INV-9) | PASS |

## κ1 / κ2 — operator gates

| Capability | Evidence | Verdict |
|---|---|---|
| a pure human gate is expressible | `docs/task-authoring.md` §5: `task_kind='deterministic'` + `always_escalates=True` + no `before_done` = pure gate, born-at-L2, task `blocked` until `resume`. §6 blesses a milestone on such a task | PASS |
| both project configs exist at the canonical path | `/home/leo/src/dark-factory/dark-factory-orchestrator.yaml` (`project_id: "dark_factory"`) and `/home/leo/src/reify/dark-factory-orchestrator.yaml` (`project_id: "reify"`) | PASS |
| the block is hot-reloadable | producer **α** via `RELOADABLE_FIELDS` (S2), upstream | PASS (DAG-direction) |
| the gate predicate has one home | `pairs >= n_min` is rendered by the report itself as the `UNDERPOWERED` tag (contract C5, producers θ1/θ2). κ2 points at it rather than re-implementing it (INV-9 `one-fact-one-home`) | PASS |

**Why κ is split.** The PRD's single κ bundles two human actions weeks apart — enable the
campaign, then rule esc-3637-1 on its report. One task would sit `blocked` on an open L2
for the whole campaign, a hold whose exit owner cannot act yet (INV-7). A deterministic
`before_done.kind='predicate'` cannot express "escalate when the data arrives" either:
predicate mode is check-then-**done**-or-escalate, so a passing check closes the task
silently and a resolved escalation re-runs the check and re-escalates. So: κ1 is the
activation gate, κ2 a second pure gate carrying a `delayed` milestone that anchors on
κ1 going `done`.

## λ — companion corrections

| Capability | Evidence | Verdict |
|---|---|---|
| `eval-framework-revival-prd.md` decision 1 exists | verbatim under §Resolved design decisions: *"**(Leo) Keep an offline *fixed* eval** — a fixed target multiple candidates compare against — refreshed with a larger, near-HEAD task set…"* | PASS |
| `fable-architect-trial-v2-prd.md` exists and has a pointer convention | Status line reads **TERMINATED 2026-08-30**; the doc already appends blockquoted post-hoc notes under Status — λ follows that convention | PASS |
| `OPERATIONS.md` has a section to slot into | 16 top-level `##` headings, numbered `1.`–`14.` plus `6a.` and `See also`. A new §"Shadow eval" after `## 7. Model routing` costs a renumbering of `8.`–`14.`, **or** takes an `a`-suffix (`## 7a.`) as `6a.` already does — λ's brief prefers the suffix, which changes one line instead of eight | PASS |
| the corpora have a place for the marker | real paths are `orchestrator/src/orchestrator/evals/tasks/` (22 fixtures, **no README**) and `…/evals/tasks_hard_v2/` (42 fixtures, `README.md` + generated `CURATION.md` + `_meta/`). The hard-v2 README already carries a `| Path | What is it |` table and already says it is not the standing corpus — the retirement marker is a row there, not a new file | PASS |
| no contract is restated | λ writes dated pointers only (INV-9 `one-fact-one-home`, INV-5) | PASS (producer) |
